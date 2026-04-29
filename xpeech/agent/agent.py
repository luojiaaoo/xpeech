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
)
from pydantic_ai.messages import ModelMessage
from ..model import ModelWrapper
from pydantic import BaseModel
from typing import AsyncGenerator
from typing import Callable, Literal
from ..utils.async_util import ensure_async
from dataclasses import dataclass
from ..utils.token_util import estimate_pydantic_ai_tokens
from .compress.summary_agent import create_summary


class MissingAgentError(Exception):
    """Raised when the agent has not been initialized before calling run."""


@dataclass
class MessageHistoryCalls:
    set_message_history: Callable[[list[ModelMessage]], None]
    get_message_history: Callable[[], list[ModelMessage]]


class AgentWrapper[T]:
    def __init__(
        self,
        model_wrapper: ModelWrapper,  # 模型包装
        deps_type: type,  # 依赖
        system_prompt: str,  # 系统提示词
        message_history_calls: MessageHistoryCalls,  # 消息历史调用
        top_p: float = 0.5,  # top_p参数
        thinking: Literal[
            "minimal", "low", "medium", "high", "xhigh"
        ] = "medium",  # 思考级别
        max_tokens: int = 8192,  # 最大响应token数
        summary_tokens: int = 8192,  # 历史消息数量超过该阈值时进行总结
        percent_summary: int = 70,  # 按照百分比去选取需要压缩历史消息
        context_window: int = 200000,  # 上下文窗口token数
    ):
        self.message_history_calls = message_history_calls
        self.summary_tokens = summary_tokens
        self.percent_summary = percent_summary
        self.context_window = context_window
        self.agent = Agent[deps_type, str](
            model_wrapper.model,
            deps_type=deps_type,
            system_prompt=system_prompt,
            model_settings=ModelSettings(
                top_p=top_p,
                parallel_tool_calls=True,
                thinking=thinking,
                max_tokens=max_tokens,
            ),
            history_processors=[
                self.context_light_processor,
                self.context_summary_processor,
            ],
        )

    def need_compress(self, total_tokens: int) -> bool:
        # 预留出summary_tokens和压缩工具上下文的空间，超过则需要压缩历史消息
        max_history_tokens = self.context_window - self.summary_tokens - 10000
        return total_tokens > max_history_tokens

    def context_light_processor(
        self,
        ctx: RunContext[T],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        total_tokens = estimate_pydantic_ai_tokens(messages)
        if self.need_compress(total_tokens):
            return messages
        else:
            return messages

    def context_summary_processor(
        self,
        ctx: RunContext[T],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        total_tokens = estimate_pydantic_ai_tokens(messages)
        if self.need_compress(total_tokens):
            num_summary_messages = int(len(messages) * self.percent_summary / 100)
            summary = create_summary(self.agent.model, messages[:num_summary_messages])
            return [*summary, *messages[num_summary_messages:]]
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
        message_history = await ensure_async(
            self.message_history_calls.get_message_history
        )()
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
                    run.all_messages()
                )
