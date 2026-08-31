import asyncio
import json
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from ..agent.server.schema import InputText
from ..config.settings import settings
from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import LLMResponse, ProviderChatKwargs, ToolCallRequest
from ..utils.helper import token_counter
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
from .tool_executor import ToolExecutor
from .tools.question import USER_TIMEOUT


@dataclass
class QuestionEvent:
    """保存等待用户回答时使用的事件和答案。"""

    event: asyncio.Event
    answer: str = "User timeout answering the question"


class AgentLoop:
    """Agent循环处理逻辑。"""

    ITERATION_STOP_PROMPT = "You have reached the maximum number of iterations and MUST stop calling tools."
    SESSION_QUESTION_EVENT: ClassVar[dict[str, QuestionEvent]] = {}

    def __init__(
        self,
        provider: LiteLLMProvider,
        workspace: Path,
        tools: list[str],
        summary_tokens: int = 8192,
        max_iterations: int = 40,
        provider_chat_kwargs: ProviderChatKwargs | None = None,
    ):
        """初始化模型提供方、会话组件和循环配置。"""
        self.provider = provider
        self.workspace = workspace
        self.tools = tools
        self.summary_tokens = summary_tokens
        self.provider_chat_kwargs = {} if provider_chat_kwargs is None else provider_chat_kwargs.to_dict()
        self.max_iterations = max_iterations
        self.max_accept_token = int(self.provider.default_context_token * 0.9 - self.summary_tokens)
        self.history = YamlHistoryRepository(settings.path.session_history_path)
        self.records = SqliteConversationRecordRepository()
        self._model_call_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self.tool_executor = ToolExecutor(
            workspace=self.workspace,
            max_result_chars=settings.tool.max_result_chars,
        )
        self.compressor = ConversationCompressor(
            chat=self.chat,
            summary_tokens=self.summary_tokens,
            max_accept_tokens=self.max_accept_token,
            target_tokens=int(self.provider.default_context_token * 0.4),
        )
        self.memory_consolidator = MemoryConsolidator(
            store=MemoryStore(workspace=self.workspace),
            chat=self.chat,
            execute_tools=self.tool_executor.execute,
            summary_tokens=self.summary_tokens,
        )

    async def tool_call(self, response: LLMResponse, messages_yaml: list, loop_count: int, session_id: str):
        """执行模型发起的工具调用，并逐步产生工具相关事件。"""
        logger.info(
            "Processing tool calls loop_count={} count={}",
            loop_count,
            len(response.tool_calls),
        )

        # 输出工具调用内容
        if response.content and response.content.strip():
            yield {"event": "assistant", "context": response.content}

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
        tool_call_result = []
        with_metas = []
        execution_results = await self.tool_executor.execute(
            response.tool_calls,
            response.mapping_tool_call_funcs,
            loop_count=loop_count,
        )
        for execution in execution_results:
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

            if tool_call.name == "send_file" and success:
                # 发送文件
                logger.info("Sending file {}", result)
                yield {"event": "send_file", "context": result}
                append_tool_result_messages_yaml(tool_call, result)
            elif tool_call.name == "ask_user_question" and success:
                # 等待用户输入
                logger.info(f"Waiting for user input, timeout={USER_TIMEOUT:.0f}s")
                self.SESSION_QUESTION_EVENT[session_id] = QuestionEvent(event=asyncio.Event())
                yield {"event": "question", "context": result}
                try:
                    await asyncio.wait_for(self.SESSION_QUESTION_EVENT[session_id].event.wait(), timeout=USER_TIMEOUT)
                    logger.info("User answered the question {}", self.SESSION_QUESTION_EVENT[session_id].answer)
                except TimeoutError:
                    logger.warning("User timeout answering the question")
                append_tool_result_messages_yaml(tool_call, self.SESSION_QUESTION_EVENT[session_id].answer)
            else:
                append_tool_result_messages_yaml(tool_call, result)
                tool_call_result.append(
                    (tool_call.id, tool_call.name, result, duration_seconds)
                )

        # 如果包含_meta属性，则把这类消息转成user消息
        if with_metas:
            messages_yaml.append(
                {
                    "role": "user",
                    "content": with_metas,
                }
            )

        # 输出工具调用结果消息
        yield {"event": "tool_call_result", "context": json.dumps(tool_call_result)}

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

    # ----------------- agent loop run -----------------

    async def run(self, message: InboundMessage):
        """运行一次Agent循环，处理一次用户消息。"""
        start_time = time.perf_counter()
        logger.info("Agent run started workspace={}", self.workspace)
        messages_yaml: list[dict] = await self.history.load(message.session_id)
        # 拼接系统提示词
        messages_yaml = set_system_prompt(messages_yaml, await build_system_prompt(workspace=self.workspace))
        # 拼接用户消息（给user添加时间戳字典，用于二级压缩）
        messages_yaml.append(
            await build_user_prompt(
                message=message,
                workspace=self.workspace,
            ),
        )

        # 用户命令拦截器
        if (
            len(message.content) == 1
            and isinstance(message.content[0], InputText)
            and (command := message.content[0].text).startswith("/")
        ):
            command = command.strip()
            if command == "/help":
                yield {"event": "command", "context": "/new -> 开始一个新会话\n/clear -> 清空上下文（不进行记忆总结）"}
                return
            elif command == "/new":
                result = await self.memory_consolidator.consolidate(messages_yaml[:-1])
                await self.history.delete(message.session_id)
                yield {"event": "command", "context": f"新会话, {result.message}"}
                return
            elif command == "/clear":
                await self.history.delete(message.session_id)
                yield {"event": "command", "context": "上下文已清空"}
                return
            yield {
                "event": "command",
                "context": f"Oops! I don't recognize {command}. Try entering /help for a list of commands.",
            }
            return

        final_content = None
        loop_count = -1
        for loop_count in count():
            if self.max_iterations is not None and loop_count >= self.max_iterations:
                final_content = (
                    f"Agent loop has reached the maximum number of iterations({self.max_iterations}) and stop."
                )
                logger.warning(
                    "Agent loop reached max iterations loop_count={} max_iterations={}",
                    loop_count,
                    self.max_iterations,
                )
                break

            # 判断是否需要压缩
            if await self.compressor.should_compress(messages_yaml):
                await self.memory_consolidator.consolidate(messages_yaml)
                messages_yaml = await self.compressor.compress(messages_yaml)

            logger.info("Calling provider chat loop_count={}", loop_count)

            try:
                response = await self.chat(
                    messages=messages_yaml,
                    tools=self.tools,
                    **self.provider_chat_kwargs,
                )
            except Exception:
                logger.exception(
                    "Provider chat failed loop_count={}",
                    loop_count,
                )
                raise
            logger.info(
                "Provider chat completed loop_count={} has_tool_calls={}",
                loop_count,
                response.has_tool_calls,
            )

            # 输出思考内容
            if response.reasoning_content and response.reasoning_content.strip():
                yield {"event": "thinking", "context": response.reasoning_content}

            # 如果有工具调用
            if response.has_tool_calls:
                async for i in self.tool_call(response, messages_yaml, loop_count, message.session_id):
                    yield i

            else:
                # 没有工具，结束循环
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
            logger.warning("Agent loop finished without final content")

        # 拼接助手消息
        messages_yaml.append({"role": "assistant", "content": final_content})

        # 输出助手消息
        yield {"event": "assistant", "context": final_content}

        # 保存历史记录
        await self.history.save(message.session_id, messages_yaml)

        # 追加本轮对话记录
        user_question = "\n".join(
            input_content.text for input_content in message.content if isinstance(input_content, InputText)
        )
        duration_s = time.perf_counter() - start_time
        await self.records.append(
            ConversationRecord(
                session_id=message.session_id,
                sender_name=message.sender_name,
                user_question=user_question,
                model_response=final_content,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                model_call_count=self._model_call_count,
                duration_s=duration_s,
            ),
        )

        # 输出token使用百分比
        token_count = await token_counter(messages_yaml)
        token_percent = min(1, token_count / self.max_accept_token) * 100
        token_notify = "♻️ 即将达到最大令牌数，强制压缩" if token_percent > 90 else ""
        yield {
            "event": "token_usage",
            "context": json.dumps(
                {
                    "上下文使用率": f"{token_percent:.2f}% ({token_count // 1000}k / {self.max_accept_token // 1000}k) {token_notify}".strip(),
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
        prompt_tokens = response.usage.get("prompt_tokens")
        completion_tokens = response.usage.get("completion_tokens")
        if prompt_tokens is None:
            logger.warning("Provider response missing prompt_tokens")
        if completion_tokens is None:
            logger.warning("Provider response missing completion_tokens")
        self._input_tokens += prompt_tokens or 0
        self._output_tokens += completion_tokens or 0
        return response
