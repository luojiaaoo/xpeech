import asyncio
import json
import time
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, ClassVar

import anyio
from fastapi import HTTPException, status
from loguru import logger

from ..config.settings import settings
from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import LLMResponse, ToolCallRequest
from ..utils.database import get_async_database_engine
from ..utils.helper import token_counter
from ..utils.question import USER_TIMEOUT, build_execution_question
from .compression import AUTOCOMPACT_BUFFER_TOKENS, ConversationCompressor
from .helper import strip_internal_message_metadata
from .history import YamlHistoryRepository
from .memory import MemoryConsolidator, MemoryStore
from .model_anomaly import ModelAnomalyGuard
from .prompt.consolidation import SUMMARY_TOKENS
from .prompt.context import build_context_messages
from .prompt.helper import (
    build_user_prompt,
    set_system_prompt,
)
from .prompt.system import build_system_prompt
from .record import ConversationRecord, ConversationRecordRepository
from .server.context import get_request_id
from .server.schema import InboundMessage, InputText
from .server.session_guard import SessionInbox
from .tool_executor import ToolExecutor
from .token_usage import TokenUsage
from .tools.execution_request import EXECUTION_DESCRIPTION, EXECUTION_PARAMETERS, EXECUTION_TITLE
from .tools.interactions import (
    ExecutionRequest,
    InteractionRequest,
    QuestionRequest,
    SendFileRequest,
)


def _append_turn_message(
    messages_yaml: list[dict[str, Any]],
    turn_messages: list[dict[str, Any]] | None,
    message: dict[str, Any],
) -> None:
    """同步追加上下文和本轮独立归档，归档不受后续上下文压缩影响。"""
    messages_yaml.append(message)
    if turn_messages is not None:
        turn_messages.append(deepcopy(message))


@dataclass
class QuestionEvent:
    """保存等待用户回答时使用的事件和答案。"""

    event: asyncio.Event
    answer: str = "User timeout answering the question"


class AgentLoop:
    """Agent循环处理逻辑。"""

    SESSION_QUESTION_EVENT: ClassVar[dict[tuple[str, str], QuestionEvent]] = {}

    def __init__(
        self,
        provider: LiteLLMProvider,
        workspace: Path,
        tools: list[str],
        channel: str | None = None,
        user_organization: str | None = None,
        mapping_tool_call_funcs: dict[str, Any] | None = None,
    ):
        """初始化模型提供方、会话组件和循环配置。"""
        self.provider = provider
        self.workspace = workspace
        self.tools = tools
        self.channel = channel
        self.user_organization = user_organization
        self.mapping_tool_call_funcs = mapping_tool_call_funcs if mapping_tool_call_funcs is not None else {}
        # 避免 token 预估和实际有差距，按模型窗口的 90% 扣除摘要预算和固定 buffer。
        self.max_accept_token = (
            int(self.provider.default_context_token * 0.9) - SUMMARY_TOKENS - AUTOCOMPACT_BUFFER_TOKENS
        )
        self.history = YamlHistoryRepository(
            settings.path.session_history_path,
            max_history_days=settings.llm.max_history_days,
        )
        self.records = ConversationRecordRepository(get_async_database_engine())
        self._model_call_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self.tool_executor = ToolExecutor(
            workspace=self.workspace,
            max_result_chars=settings.tool.max_result_chars,
            mapping_tool_call_funcs=self.mapping_tool_call_funcs,
            max_tool_timeout=settings.tool.max_tool_timeout,
        )
        self.compressor = ConversationCompressor(
            chat=self.chat,
            max_accept_tokens=self.max_accept_token,
            target_tokens=self.max_accept_token // 2,
            token_counter=self._count_context_tokens,
        )
        self.memory_consolidator = MemoryConsolidator(
            store=MemoryStore(workspace=self.workspace),
            chat=self.chat,
            execute_tools=self.tool_executor.execute,
        )

    async def tool_call(
        self,
        response: LLMResponse,
        messages_yaml: list,
        loop_count: int,
        user_id: str,
        session_id: str,
        sender_name: str,
        session_metadata: dict[str, str] | None = None,
        channel: str | None = None,
        turn_messages: list[dict[str, Any]] | None = None,
    ):
        """执行模型发起的工具调用，并逐步产生工具相关事件。"""
        logger.info(
            "Processing tool calls loop_count={} count={}",
            loop_count,
            len(response.tool_calls),
        )

        # 还原工具调用格式
        tool_call_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in response.tool_calls
        ]

        # 创建助手消息
        msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        if tool_call_dicts:
            msg["tool_calls"] = tool_call_dicts
        _append_turn_message(messages_yaml, turn_messages, msg)

        # 先记录，再输出；暂停在 yield 时，finally 仍能保存这条消息。
        yield {
            "event": "tool_call",
            "context": json.dumps([
                {"tool_call_id": i.id, "name": i.name, "arguments": i.arguments}
                for i in response.tool_calls
            ]),
        }

        # 执行工具调用
        with_metas = []

        def prepend_text(content: Any, text: str) -> Any:
            if isinstance(content, str):
                return f"{text}\n\n{content}"
            if isinstance(content, list):
                return [{"type": "text", "text": text}, *content]
            return content

        def append_tool_result_messages_yaml(
            tool_call: ToolCallRequest, result_: Any, *, started_at: datetime,
            duration_seconds: float,
        ):
            """将一次工具调用结果转换为对话历史消息。"""
            timing = f"【工具调用】发起时间：{started_at:%H:%M:%S}；耗时：{duration_seconds:.2f} 秒"
            result_ = prepend_text(result_, timing)
            # 创建工具调用结果消息
            if isinstance(result_, str):
                _append_turn_message(
                    messages_yaml, turn_messages,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_,
                    }
                )
            elif isinstance(result_, list):
                # 把带_meta属性的字典过滤出来
                others = []
                for i in result_:
                    if isinstance(i, dict) and "_meta" in i:
                        # 提取_meta属性，组装成消息
                        _meta = i.pop("_meta")
                        with_metas.extend([{"type": "text", "text": _meta}, i])
                    else:
                        others.append(i)
                _append_turn_message(
                    messages_yaml, turn_messages,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": others,
                    }
                )

        async with aclosing(self.tool_executor.execute(
            response.tool_calls,
            loop_count=loop_count,
            session_metadata=session_metadata,
            user_id=user_id,
            sender_name=sender_name,
            channel=channel,
            session_id=session_id,
        )) as executions:
            async for execution in executions:
                tool_call = execution.call
                result = execution.value
                success = execution.succeeded
                duration_seconds = execution.duration_seconds

                if isinstance(execution.value, InteractionRequest) and success:
                    if isinstance(execution.value, ExecutionRequest):
                        execution_function = self.mapping_tool_call_funcs[tool_call.name].execution_function
                        arguments = {key: value for key, value in tool_call.arguments.items() if key not in EXECUTION_PARAMETERS}
                        question = build_execution_question(
                            tool_call.name, arguments, tool_call.arguments[EXECUTION_DESCRIPTION],
                            tool_call_id=tool_call.id,
                            title=tool_call.arguments.get(EXECUTION_TITLE),
                        )
                        # 等待用户确认本次执行。
                        logger.info(f"Waiting for user input, timeout={USER_TIMEOUT:.0f}s")
                        session_key = (user_id, session_id)
                        question_event = QuestionEvent(event=asyncio.Event())
                        self.SESSION_QUESTION_EVENT[session_key] = question_event
                        question_start_time = time.perf_counter()
                        timed_out = False
                        try:
                            yield {"event": "question", "context": question}
                            await asyncio.wait_for(question_event.event.wait(), timeout=USER_TIMEOUT)
                            logger.info("User answered the question {}", question_event.answer)
                        except TimeoutError:
                            timed_out = True
                            logger.warning("User timeout answering the question")
                        finally:
                            self.SESSION_QUESTION_EVENT.pop(session_key, None)
                        # 覆盖方法调用的时间，遵守实际用户回答耗时
                        duration_seconds = time.perf_counter() - question_start_time
                        decision = question_event.answer
                        user_confirmed = not timed_out and "confirm" in decision and "cancel" not in decision
                        if user_confirmed:
                            execution_result = await self.tool_executor.execute_function(
                                tool_call,
                                execution_function,
                                arguments,
                                loop_count=loop_count,
                                session_metadata=session_metadata,
                                user_id=user_id,
                                sender_name=sender_name,
                                channel=channel,
                                session_id=session_id,
                            )
                            result = execution_result.value
                            duration_seconds += execution_result.duration_seconds
                            success = execution_result.succeeded
                            notice = "【用户确认后执行】用户已同意本次操作，工具随后才开始执行。以下为实际结果："
                            result = prepend_text(result, notice)
                        elif timed_out:
                            success = False
                            result = "用户确认超时，工具未执行。"
                        elif "cancel" in decision:
                            success = False
                            result = "用户取消执行，工具未执行。"
                        else:
                            success = False
                            result = "执行确认回复格式异常，工具未执行。"

                        append_tool_result_messages_yaml(
                            tool_call, result, started_at=execution.started_at,
                            duration_seconds=duration_seconds,
                        )
                        tool_result_content = result
                    elif isinstance(execution.value, SendFileRequest):
                        # 发送文件
                        logger.info("Sending file {}", execution.value.source)
                        tool_result_content = f"文件已发送：{execution.value.source}"
                        append_tool_result_messages_yaml(
                            tool_call, tool_result_content, started_at=execution.started_at,
                            duration_seconds=duration_seconds,
                        )
                        yield {"event": "send_file", "context": execution.value.source}
                    elif isinstance(execution.value, QuestionRequest):
                        # 等待用户输入
                        logger.info(f"Waiting for user input, timeout={USER_TIMEOUT:.0f}s")
                        session_key = (user_id, session_id)
                        question_event = QuestionEvent(event=asyncio.Event())
                        self.SESSION_QUESTION_EVENT[session_key] = question_event
                        question_start_time = time.perf_counter()
                        try:
                            yield {"event": "question", "context": execution.value.question}
                            await asyncio.wait_for(question_event.event.wait(), timeout=USER_TIMEOUT)
                            logger.info("User answered the question {}", question_event.answer)
                        except TimeoutError:
                            success = False
                            logger.warning("User timeout answering the question")
                        finally:
                            self.SESSION_QUESTION_EVENT.pop(session_key, None)
                        # 覆盖方法调用的时间，遵守实际用户回答耗时
                        duration_seconds = time.perf_counter() - question_start_time
                        answer = question_event.answer
                        append_tool_result_messages_yaml(
                            tool_call, answer, started_at=execution.started_at,
                            duration_seconds=duration_seconds,
                        )
                        tool_result_content = answer
                    else:
                        raise TypeError(
                            f"未处理的交互请求类型: {type(execution.value).__name__} (tool={tool_call.name})"
                        )
                else:
                    append_tool_result_messages_yaml(
                        tool_call, result, started_at=execution.started_at,
                        duration_seconds=duration_seconds,
                    )
                    tool_result_content = result

                tool_call_result = {
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "result": tool_result_content,
                    "duration_seconds": duration_seconds,
                    "success": success,
                }
                yield {"event": "tool_call_result", "context": json.dumps(tool_call_result)}

        # 如果包含_meta属性，则把这类消息转成user消息
        if with_metas:
            _append_turn_message(
                messages_yaml, turn_messages,
                {
                    "role": "user",
                    "content": with_metas,
                }
            )

    # ----------------- agent loop run -----------------

    async def run(
        self,
        message: InboundMessage,
        *,
        not_interactive: bool = False,
        inbox: SessionInbox | None = None,
    ):
        """运行 Agent 循环，在无工具调用的迭代结束时接收追加消息。"""
        start_time = time.perf_counter()
        request_id = get_request_id()
        if request_id is None:
            raise RuntimeError("Agent run requires a request_id")
        logger.info("Agent run started workspace={}", self.workspace)

        # 用户命令拦截器
        if not not_interactive and (command := message.command_text()) is not None:
            command_parts = command.split(maxsplit=1)
            command_name = command_parts[0]
            continuation = command_parts[1].strip() if len(command_parts) == 2 else ""
            if command_name == "/help":
                yield {"event": "command", "context": "/new -> 开始一个新会话\n/clear -> 清空上下文（不进行记忆总结）"}
                return
            elif command_name == "/new":
                history = await self.history.load(message.user_id, message.session_id, channel=message.channel)
                result = await self.memory_consolidator.consolidate(history)
                await self.history.delete(message.user_id, message.session_id, channel=message.channel)
                yield {"event": "command", "context": f"新会话, {result.message}"}
                if not continuation:
                    return
            elif command_name == "/clear":
                await self.history.delete(message.user_id, message.session_id, channel=message.channel)
                yield {"event": "command", "context": "上下文已清空"}
                if not continuation:
                    return
            else:
                yield {
                    "event": "command",
                    "context": f"Oops! I don't recognize {command}. Try entering /help for a list of commands.",
                }
                return

            # 将命令后的内容作为清空上下文后的第一条用户消息。
            message.content[0].text = continuation

        messages_yaml: list[dict] = (
            []
            if not_interactive
            else await self.history.load(message.user_id, message.session_id, channel=message.channel)
        )

        # 固定系统提示词后按类型插入本轮最新的动态 user 上下文。
        system_prompt = await build_system_prompt(channel=self.channel)
        token_usage = TokenUsage(
            provider=self.provider,
            tools=self.tools,
            compression_threshold=self.max_accept_token,
            context_tokens={"系统提示词": await token_counter([system_prompt])},
        )
        messages_yaml = set_system_prompt(
            messages_yaml,
            system_prompt,
            await build_context_messages(
                workspace=self.workspace,
                user_id=message.user_id,
                channel=self.channel,
                user_organization=self.user_organization,
                not_interactive=not_interactive,
                token_counts=token_usage.context_tokens,
            ),
        )
        # 时间戳标记真实用户请求，压缩时按完整回合切分。
        turn_messages: list[dict[str, Any]] = []

        _append_turn_message(
            messages_yaml, turn_messages,
            await build_user_prompt(message=message, workspace=self.workspace),
        )

        try:
            if inbox is not None:
                inbox.start()
            final_content = None
            final_content_error = False
            anomaly_guard = ModelAnomalyGuard()
            for loop_count in count():
                # 判断是否需要压缩
                if await self.compressor.should_compress(messages_yaml):
                    yield {"event": "command", "context": "♻️ 压缩上下文中，请稍等..."}
                    await self.memory_consolidator.consolidate(messages_yaml)
                    messages_yaml = await self.compressor.compress(messages_yaml)

                logger.info("Calling provider chat loop_count={}", loop_count)

                try:
                    response = await self.chat(
                        messages=anomaly_guard.prepare_messages(messages_yaml),
                        tools=self.tools,
                        json_output=message.json_output,
                    )

                    async with aclosing(response.iter_mix_chunks) as chunks:
                        async for kind, chunk in chunks:
                            if kind == "reasoning_content" and isinstance(chunk, str):
                                yield {"event": "thinking", "context": chunk}
                            elif kind == "reasoning_content_end":
                                yield {"event": "thinking_end", "context": ""}
                            elif kind == "content" and isinstance(chunk, str):
                                yield {"event": "assistant", "context": chunk}
                            elif kind == "content_end":
                                yield {"event": "assistant_end", "context": ""}
                            elif (kind == "tool_calls" and isinstance(chunk, ToolCallRequest)) or (kind == "tool_calls_end"):
                                pass  # 工具调用事件在后续处理
                except HTTPException:
                    raise
                except Exception as exc:
                    logger.exception(
                        "Provider chat failed loop_count={}",
                        loop_count,
                    )
                    _append_turn_message(
                        messages_yaml, turn_messages,
                        {
                            "role": "user",
                            "content": (
                                f"The previous model call failed with this error: {exc}. "
                                "Please retry the request and adjust your approach if needed."
                            ),
                        }
                    )
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(exc),
                    ) from exc

                logger.info(
                    "Provider chat completed loop_count={} has_tool_calls={}",
                    loop_count,
                    response.has_tool_calls,
                )

                # 如果有工具调用
                if response.has_tool_calls:
                    async with aclosing(self.tool_call(
                        response,
                        messages_yaml,
                        loop_count,
                        message.user_id,
                        message.session_id,
                        message.sender_name,
                        session_metadata=message.session_metadata,
                        channel=message.channel,
                        turn_messages=turn_messages,
                    )) as tool_events:
                        async for i in tool_events:
                            yield i
                else:
                    final_content = response.content
                    pending = inbox.drain_or_close() if inbox is not None else []
                    if not pending:
                        break

                    # 先记录当前回复，再逐条写入累计的用户消息。
                    if final_content is not None:
                        _append_turn_message(
                            messages_yaml, turn_messages, {"role": "assistant", "content": final_content},
                        )
                    for incoming in pending:
                        _append_turn_message(
                            messages_yaml, turn_messages,
                            await build_user_prompt(message=incoming, workspace=self.workspace),
                        )

                anomaly_guard.observe_tool_calls(response.tool_calls)

            if final_content is None:
                final_content = "I've completed processing but have no response to give."
                final_content_error = True
                logger.warning("Agent loop finished without final content")

            # 先记录助手消息，再发送异常兜底内容。
            _append_turn_message(messages_yaml, turn_messages, {"role": "assistant", "content": final_content})

            if final_content_error:
                yield {"event": "assistant", "context": final_content}
                yield {"event": "assistant_end", "context": ""}

        finally:
            # 取消请求时也保存已经加入上下文的消息。
            with anyio.CancelScope(shield=True):
                if inbox is not None:
                    inbox.close()
                    # 停止或异常时保留已接收但还没有处理的消息。
                    for incoming in inbox.drain():
                        _append_turn_message(
                            messages_yaml, turn_messages,
                            await build_user_prompt(message=incoming, workspace=self.workspace),
                        )
                await self.history.save(
                    message.user_id,
                    message.session_id,
                    messages_yaml,
                    channel=message.channel,
                )

        # 整个 loop 完成后只记录一次；中途入队的请求不单独创建 record。
        user_question = "\n".join(
            input_content.text for input_content in message.content if isinstance(input_content, InputText)
        )
        duration_s = time.perf_counter() - start_time
        record = ConversationRecord(
            user_id=message.user_id,
            session_id=message.session_id,
            request_id=request_id,
            channel=message.channel,
            sender_name=message.sender_name,
            user_question=user_question,
            model_response=final_content,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model_call_count=self._model_call_count,
            duration_s=duration_s,
        )
        await self.records.append(record, messages=turn_messages)
        yield await token_usage.build_event(
            messages_yaml,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model_call_count=self._model_call_count,
            duration_s=duration_s,
        )

        logger.info("Agent run completed")

    async def _count_context_tokens(self, messages: list[dict[str, Any]]) -> int:
        """按原有整包估算判断是否需要压缩，与展示用的分项统计独立。"""
        return await self.provider.chat_count_tokens(messages=strip_internal_message_metadata(messages), tools=self.tools)

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> LLMResponse:
        """移除内部元数据后调用模型提供方。"""
        response = await self.provider.chat(messages=strip_internal_message_metadata(messages), **kwargs)
        self._model_call_count += 1
        response.add_completion_callback(self._record_usage)
        return response

    def _record_usage(self, response: LLMResponse) -> None:
        prompt_tokens = response.usage.get("prompt_tokens")
        completion_tokens = response.usage.get("completion_tokens")
        if prompt_tokens is None:
            logger.warning("Provider response missing prompt_tokens")
        if completion_tokens is None:
            logger.warning("Provider response missing completion_tokens")
        self._input_tokens += prompt_tokens or 0
        self._output_tokens += completion_tokens or 0
