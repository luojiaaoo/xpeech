import asyncio
import json
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, ClassVar

from fastapi import HTTPException, status
from loguru import logger

from ..agent.server.schema import InputText
from ..config.settings import settings
from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import LLMResponse, ToolCallRequest
from ..utils.helper import format_exception2llm, token_counter
from ..utils.question import USER_TIMEOUT, build_execution_question
from .compression import ConversationCompressor
from .helper import strip_internal_message_metadata
from .history import YamlHistoryRepository
from .memory import MemoryConsolidator, MemoryStore
from .prompt.helper import (
    build_user_prompt,
    set_system_prompt,
)
from .prompt.system import build_system_prompt
from .record import ConversationRecord, SqliteConversationRecordRepository
from .server.schema import InboundMessage
from .tool_executor import TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS, ToolExecutor
from .tools.execution_request import (
    EXECUTION_DESCRIPTION,
    EXECUTION_REQUEST,
)
from .tools.helper import get_tool_model_cls


@dataclass
class QuestionEvent:
    """保存等待用户回答时使用的事件和答案。"""

    event: asyncio.Event
    answer: str = "User timeout answering the question"


class AgentLoop:
    """Agent循环处理逻辑。"""

    ITERATION_STOP_PROMPT = "You have reached the maximum number of iterations and MUST stop calling tools."
    SESSION_QUESTION_EVENT: ClassVar[dict[tuple[str, str], QuestionEvent]] = {}

    def __init__(
        self,
        provider: LiteLLMProvider,
        workspace: Path,
        tools: list[str],
        summary_tokens: int,
        max_iterations: int,
        channel_tag: str | None = None,
    ):
        """初始化模型提供方、会话组件和循环配置。"""
        self.provider = provider
        self.workspace = workspace
        self.tools = tools
        self.summary_tokens = summary_tokens
        self.max_iterations = max_iterations
        self.channel_tag = channel_tag
        self.max_accept_token = int(self.provider.default_context_token * 0.9 - self.summary_tokens)
        self.history = YamlHistoryRepository(
            settings.path.session_history_path,
            max_files_per_user=settings.llm.max_history_files_per_user,
        )
        self.records = SqliteConversationRecordRepository()
        self._model_call_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self.tool_executor = ToolExecutor(
            workspace=self.workspace,
            max_result_chars=settings.tool.max_result_chars,
            max_tool_timeout=settings.tool.max_tool_timeout,
        )
        self.compressor = ConversationCompressor(
            chat=self.chat,
            summary_tokens=self.summary_tokens,
            max_accept_tokens=self.max_accept_token,
            target_tokens=int(self.provider.default_context_token * 0.25),
        )
        self.memory_consolidator = MemoryConsolidator(
            store=MemoryStore(workspace=self.workspace),
            chat=self.chat,
            execute_tools=self.tool_executor.execute,
            summary_tokens=self.summary_tokens,
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
    ):
        """执行模型发起的工具调用，并逐步产生工具相关事件。"""
        logger.info(
            "Processing tool calls loop_count={} count={}",
            loop_count,
            len(response.tool_calls),
        )

        # 输出工具调用消息
        yield {
            "event": "tool_call",
            "context": json.dumps([(i.id, i.name, i.arguments) for i in response.tool_calls]),
        }

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
        messages_yaml.append(msg)

        # 执行工具调用
        with_metas = []
        async for execution in self.tool_executor.execute(
            response.tool_calls,
            response.mapping_tool_call_funcs,
            loop_count=loop_count,
            session_metadata=session_metadata,
            user_id=user_id,
            sender_name=sender_name,
            channel=channel,
            session_id=session_id,
        ):
            tool_call = execution.call
            result = execution.value
            success = execution.succeeded
            duration_seconds = execution.duration_seconds

            def append_tool_result_messages_yaml(tool_call: ToolCallRequest, result_: Any):
                """将一次工具调用结果转换为对话历史消息。"""
                # 创建工具调用结果消息
                if isinstance(result_, str):
                    messages_yaml.append(
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
                    messages_yaml.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": others,
                        }
                    )

            if execution.value == EXECUTION_REQUEST and success:
                execution_function = response.mapping_tool_call_funcs[tool_call.name].execution_function
                arguments = {key: value for key, value in tool_call.arguments.items() if key != EXECUTION_DESCRIPTION}
                question = build_execution_question(
                    tool_call.name, arguments, tool_call.arguments[EXECUTION_DESCRIPTION]
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
                if not timed_out and "confirm" in decision and "cancel" not in decision:
                    execution_start_time = time.perf_counter()
                    try:
                        model_info = get_tool_model_cls(execution_function)
                        args = (model_info.model_cls(**arguments),) if model_info.has_pydantic_param else ()
                        async with asyncio.timeout(self.tool_executor._max_tool_timeout):
                            result = await execution_function(*args)
                            if isinstance(result, str) and tool_call.name not in TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS:
                                result = await self.tool_executor._limit_text(
                                    tool_call, result, self.tool_executor._max_result_chars
                                )
                    except Exception as exc:  # noqa: BLE001 - tool failures are returned to the model
                        success = False
                        result = format_exception2llm(exc)
                    duration_seconds += time.perf_counter() - execution_start_time
                elif timed_out:
                    success = False
                    result = "用户确认超时，工具未执行。"
                elif "cancel" in decision:
                    success = False
                    result = "用户取消执行，工具未执行。"
                else:
                    success = False
                    result = "执行确认回复格式异常，工具未执行。"

                append_tool_result_messages_yaml(tool_call, result)
                tool_result_content = result

            elif tool_call.name == "send_file" and success:
                # 发送文件
                logger.info("Sending file {}", result)
                yield {"event": "send_file", "context": result}
                append_tool_result_messages_yaml(tool_call, result)
                tool_result_content = f"已发送：{result}"
            elif tool_call.name == "ask_user_question" and success:
                # 等待用户输入
                logger.info(f"Waiting for user input, timeout={USER_TIMEOUT:.0f}s")
                session_key = (user_id, session_id)
                question_event = QuestionEvent(event=asyncio.Event())
                self.SESSION_QUESTION_EVENT[session_key] = question_event
                question_start_time = time.perf_counter()
                yield {"event": "question", "context": result}
                try:
                    await asyncio.wait_for(question_event.event.wait(), timeout=USER_TIMEOUT)
                    logger.info("User answered the question {}", question_event.answer)
                except TimeoutError:
                    logger.warning("User timeout answering the question")
                finally:
                    self.SESSION_QUESTION_EVENT.pop(session_key, None)
                # 覆盖方法调用的时间，遵守实际用户回答耗时
                duration_seconds = time.perf_counter() - question_start_time
                answer = question_event.answer
                append_tool_result_messages_yaml(tool_call, answer)
                tool_result_content = answer
            else:
                append_tool_result_messages_yaml(tool_call, result)
                tool_result_content = result

            tool_call_result = {
                "id": tool_call.id,
                "name": tool_call.name,
                "result": tool_result_content,
                "duration_seconds": duration_seconds,
            }
            yield {"event": "tool_call_result", "context": json.dumps(tool_call_result)}

        # 如果包含_meta属性，则把这类消息转成user消息
        if with_metas:
            messages_yaml.append(
                {
                    "role": "user",
                    "content": with_metas,
                }
            )

    # ----------------- agent loop run -----------------

    async def run(self, message: InboundMessage, *, not_interactive: bool = False):
        """运行一次Agent循环，处理一次用户消息。"""
        start_time = time.perf_counter()
        logger.info("Agent run started workspace={}", self.workspace)

        # 用户命令拦截器
        if (
            not not_interactive
            and len(message.content) == 1
            and isinstance(message.content[0], InputText)
            and (command := message.content[0].text.strip()).startswith("/")
        ):
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

        # 拼接系统提示词
        messages_yaml = set_system_prompt(
            messages_yaml,
            await build_system_prompt(
                workspace=self.workspace,
                channel_tag=self.channel_tag,
                not_interactive=not_interactive,
            ),
        )
        # 拼接用户消息（给user添加时间戳字典，用于二级压缩）
        messages_yaml.append(
            await build_user_prompt(
                message=message,
                workspace=self.workspace,
            ),
        )

        final_content = None
        final_content_error = False
        for loop_count in count():
            # 达到最大迭代次数，强制停止循环
            if self.max_iterations is not None and loop_count >= self.max_iterations:
                final_content = (
                    f"Agent loop has reached the maximum number of iterations({self.max_iterations}) and stop."
                )
                final_content_error = True
                logger.warning(
                    "Agent loop reached max iterations loop_count={} max_iterations={}",
                    loop_count,
                    self.max_iterations,
                )
                break

            # 判断是否需要压缩
            if await self.compressor.should_compress(messages_yaml):
                yield {"event": "command", "context": "♻️ 压缩上下文中，请稍等..."}
                await self.memory_consolidator.consolidate(messages_yaml)
                messages_yaml = await self.compressor.compress(messages_yaml)

            logger.info("Calling provider chat loop_count={}", loop_count)

            try:
                response = await self.chat(
                    messages=messages_yaml,
                    tools=self.tools,
                    json_output=message.json_output,
                )

                async for kind, chunk in response.iter_mix_chunks:
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
                messages_yaml.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous model call failed with this error: {exc}. "
                            "Please retry the request and adjust your approach if needed."
                        ),
                    }
                )
                await self.history.save(
                    message.user_id,
                    message.session_id,
                    messages_yaml,
                    channel=message.channel,
                    not_interactive=not_interactive,
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
                async for i in self.tool_call(
                    response,
                    messages_yaml,
                    loop_count,
                    message.user_id,
                    message.session_id,
                    message.sender_name,
                    session_metadata=message.session_metadata,
                    channel=message.channel,
                ):
                    yield i
            else:
                # 没有工具，结束循环
                final_content = response.content
                break

            # 即将达到最大迭代次数，添加用户消息，提示达到最大迭代次数
            if self.max_iterations is not None and loop_count == self.max_iterations - 2:
                logger.warning(
                    "Approaching max iterations loop_count={} max_iterations={}",
                    loop_count,
                    self.max_iterations,
                )
                messages_yaml.append(
                    {
                        "role": "user",
                        "content": self.ITERATION_STOP_PROMPT,
                    }
                )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
            final_content_error = True
            logger.warning("Agent loop finished without final content")

        # 发送异常兜底内容
        if final_content_error:
            yield {"event": "assistant", "context": final_content}

        # 拼接助手消息
        messages_yaml.append({"role": "assistant", "content": final_content})

        # 保存历史记录
        await self.history.save(
            message.user_id,
            message.session_id,
            messages_yaml,
            channel=message.channel,
            not_interactive=not_interactive,
        )

        # 追加本轮对话记录
        user_question = "\n".join(
            input_content.text for input_content in message.content if isinstance(input_content, InputText)
        )
        duration_s = time.perf_counter() - start_time
        await self.records.append(
            ConversationRecord(
                user_id=message.user_id,
                sender_name=message.sender_name,
                user_question=user_question,
                model_response=final_content,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                model_call_count=self._model_call_count,
                duration_s=duration_s,
            ),
        )

        def format_token_count(value: int) -> str:
            if value < 1_000:
                return str(value)
            amount, unit = value / 1_000, "k"
            # A rounded 1000k is more naturally displayed as 1m.
            if round(amount, 2) >= 1_000:
                amount, unit = value / 1_000_000, "m"
            return f"{amount:.2f}".rstrip("0").rstrip(".") + unit

        # Context size is estimated separately from provider-reported consumption.
        token_count = await token_counter(messages_yaml)
        token_percent = min(1, token_count / self.max_accept_token) * 100
        token_notify = "♻️ 即将达到最大上下文" if token_percent > 90 else ""
        context_usage = (
            f"{token_percent:.2f}% "
            f"({format_token_count(token_count)} / {format_token_count(self.max_accept_token)}) {token_notify}"
        ).strip()
        yield {
            "event": "token_usage",
            "context": json.dumps(
                {
                    "上下文使用率": context_usage,
                    "输入 Token": format_token_count(self._input_tokens),
                    "输出 Token": format_token_count(self._output_tokens),
                    "总 Token": format_token_count(self._input_tokens + self._output_tokens),
                    "会话时长": f"{duration_s:.0f}秒",
                    "大模型请求次数": str(self._model_call_count),
                },
                ensure_ascii=False,
            ),
        }

        logger.info("Agent run completed")

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
