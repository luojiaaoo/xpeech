from pydantic_ai import (
    Agent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
    ModelSettings,
    ModelMessage,
    RunContext,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ToolReturnPart,
    ModelResponse,
)
from textwrap import dedent
import json
from pydantic_core import to_jsonable_python
from .model import ModelWrapper
from pydantic import BaseModel
from typing import AsyncGenerator
from typing import Callable, Literal
from ..utils.async_util import ensure_async
from dataclasses import dataclass
from ..utils.token_util import estimate_pydantic_ai_tokens
from .compress.summary_agent import create_summary
from pydantic_ai.capabilities import Thinking
from pathlib import Path
from .tool.filesystem import FilesystemTools
from pydantic_ai.capabilities import ThreadExecutor
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from datetime import timedelta
from loguru import logger

executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="agent-worker")


class MissingAgentError(Exception):
    """Raised when the agent has not been initialized before calling run."""


@dataclass
class MessageHistoryCalls:
    set_message_history: Callable[[str], None]
    get_message_history: Callable[[], str]


class AgentWrapper[T]:
    def __init__(
        self,
        model_wrapper: ModelWrapper,  # 模型包装
        deps_type: type,  # 依赖
        system_prompt: str,  # 系统提示词
        message_history_calls: MessageHistoryCalls,  # 消息历史调用
        top_p: float = 0.5,  # top_p参数
        thinking: Literal[
            "minimal", "low", "medium", "high", "xhigh", False
        ] = "medium",  # 思考级别
        max_tokens: int = 8192,  # 最大响应token数
        summary_tokens: int = 8192,  # 历史消息数量超过该阈值时进行总结
        percent_summary: int = 70,  # 按照百分比去选取需要压缩历史消息
        context_window: int = 200000,  # 上下文窗口token数
        workspace: Path | None = None,  # 工作空间，用于文件系统工具
    ):
        self.message_history_calls = message_history_calls
        self.summary_tokens = summary_tokens
        self.percent_summary = percent_summary
        self.context_window = context_window
        self.workspace = workspace

        self.agent = Agent[deps_type, str](
            model_wrapper.model,
            deps_type=deps_type,
            system_prompt=system_prompt,
            model_settings=ModelSettings(
                top_p=top_p,
                parallel_tool_calls=True,
                max_tokens=max_tokens,
            ),
            capabilities=[
                Thinking(effort=thinking),
                ThreadExecutor(executor),
            ],
            history_processors=[
                self._context_tool_result_budget_processor,
                self._context_tool_result_timeout_processor,
                self._context_summary_processor,
            ],
        )
        tool_parameter = dict(
            defer_loading=True,
            docstring_format="google",
            require_parameter_descriptions=True,
        )
        if self.workspace:
            self.fs_tools = FilesystemTools(workspace)
            self.agent.tool_plain(self.fs_tools.read_file, **tool_parameter)
            self.agent.tool_plain(self.fs_tools.write_file, **tool_parameter)
            self.agent.tool_plain(self.fs_tools.create_file, **tool_parameter)
            self.agent.tool_plain(self.fs_tools.delete_file, **tool_parameter)
            self.agent.tool_plain(self.fs_tools.move_file, **tool_parameter)
            self.agent.tool_plain(self.fs_tools.copy_file, **tool_parameter)
            self.agent.tool_plain(self.fs_tools.search_files, **tool_parameter)
            self.agent.tool_plain(self.fs_tools.list_dir, **tool_parameter)

    def _need_compress(self, total_tokens: int) -> bool:
        """判断是否需要压缩，预留出summary_tokens和压缩工具上下文的空间，超过则需要压缩历史消息"""
        max_history_tokens = self.context_window - self.summary_tokens - 10000
        return total_tokens > max_history_tokens

    def _context_tool_result_budget_processor(
        self,
        ctx: RunContext[T],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        """把超大的tool结果保存到文件，只返回一个提示文本，避免占用过多上下文窗口"""
        total_tokens = estimate_pydantic_ai_tokens(messages)
        if self._need_compress(total_tokens):
            for m in messages:
                if not isinstance(m, ModelRequest):
                    continue
                for part in m.parts:
                    if not isinstance(part, ToolReturnPart):
                        continue
                    content = str(part.content)
                    if len(content) > 4000:
                        save_path = f"tool_results/{uuid4().hex}.txt"
                        self.fs_tools.write_file(save_path, content)
                        part.content = dedent(f"""
                            <persisted-output>
                                Output too large ({len(content)} char). Full output saved to: {save_path}

                                Preview:
                                {content[:2000]}
                            </persisted-output>
                        """).lstrip()
            logger.info(
                f"一级压缩-将过大的tool结果保存到文件: {total_tokens} to {estimate_pydantic_ai_tokens(messages)}"
            )
            return messages
        else:
            return messages

    def _context_tool_result_timeout_processor(
        self,
        ctx: RunContext[T],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        """删除过期的tool调用内容，避免占用上下文窗口"""
        total_tokens = estimate_pydantic_ai_tokens(messages)
        if self._need_compress(total_tokens):
            timeout_hours = 1
            newest_time = messages[-1].timestamp
            for m in messages:
                if not isinstance(m, ModelRequest):
                    continue
                if m.timestamp > newest_time - timedelta(hours=timeout_hours):
                    continue
                for part in m.parts:
                    if not isinstance(part, ToolReturnPart):
                        continue
                    part.content = "[Old tool result content cleared]"
            logger.info(
                f"二级压缩-清除超过{timeout_hours}小时的tool结果: {total_tokens} to {estimate_pydantic_ai_tokens(messages)}"
            )
            return messages
        else:
            return messages

    async def _context_summary_processor(
        self,
        ctx: RunContext[T],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        """如果消息总Token数超过上下文窗口限制，则对部分历史消息进行总结压缩，保留总结后的消息和最新的消息。"""
        total_tokens = estimate_pydantic_ai_tokens(messages)
        if self._need_compress(total_tokens):
            num_summary_messages = int(len(messages) * self.percent_summary / 100)
            # 按一问一答去截断
            for i in range(
                num_summary_messages, max(num_summary_messages + 1, len(messages) - 4)
            ):
                if isinstance(messages[i], ModelResponse):
                    num_summary_messages = i
                    break
            summary = await create_summary(
                self.agent.model, messages[:num_summary_messages]
            )
            messages = [*summary, *messages[num_summary_messages:]]
            logger.info(
                f"三级压缩-总结{num_summary_messages}条历史消息: {total_tokens} to {estimate_pydantic_ai_tokens(messages)}"
            )
            return messages
        else:
            return messages

    async def run(
        self, user_prompt: str, output_type: type[BaseModel]
    ) -> AsyncGenerator:
        if self.agent is None:
            raise MissingAgentError(
                "Agent has not been initialized. Call set_agent() first."
            )
        # 获取历史消息
        message_history_json: str = await ensure_async(
            self.message_history_calls.get_message_history
        )()
        message_history: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(
            json.loads(message_history_json) if message_history_json.strip() else []
        )
        async with self.agent.iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
        ) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    # A user prompt node => The user has provided input
                    yield f"=== UserPromptNode: {node.user_prompt} ==="
                elif Agent.is_model_request_node(node):
                    # A model request node => We can stream tokens from the model's request
                    yield "=== ModelRequestNode: streaming partial request tokens ==="
                    async with node.stream(run.ctx) as request_stream:
                        final_result_found = False
                        async for event in request_stream:
                            if isinstance(event, PartStartEvent):
                                yield f"[Request] Starting part {event.index}: {event.part!r}"
                            elif isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, TextPartDelta):
                                    yield f"[Request] Part {event.index} text delta: {event.delta.content_delta!r}"
                                elif isinstance(event.delta, ThinkingPartDelta):
                                    yield f"[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}"
                                elif isinstance(event.delta, ToolCallPartDelta):
                                    yield f"[Request] Part {event.index} args delta: {event.delta.args_delta}"
                            elif isinstance(event, FinalResultEvent):
                                yield f"[Result] The model started producing a final result (tool_name={event.tool_name})"
                                final_result_found = True
                                break
                        if final_result_found:
                            # Once the final result is found, we can call `AgentStream.stream_text()` to stream the text.
                            # A similar `AgentStream.stream_output()` method is available to stream structured output.
                            async for output in request_stream.stream_text():
                                yield f"[Output] {output}"
                elif Agent.is_call_tools_node(node):
                    # A handle-response node => The model returned some data, potentially calls a tool
                    yield "=== CallToolsNode: streaming partial response & tool usage ==="
                    async with node.stream(run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                yield f"[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})"
                            elif isinstance(event, FunctionToolResultEvent):
                                yield f"[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}"
                elif Agent.is_end_node(node):
                    # Once an End node is reached, the agent run is complete
                    assert run.result is not None
                    assert run.result.output == node.data.output
                    yield f"=== Final Agent Output: {run.result.output} ==="
                # 存储对话历史
                await ensure_async(self.message_history_calls.set_message_history)(
                    json.dumps(
                        to_jsonable_python(run.all_messages()),
                        ensure_ascii=False,
                    )
                )
