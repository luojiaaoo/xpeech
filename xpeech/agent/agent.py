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
)
from ..model import ModelWrapper
from pydantic import BaseModel
from typing import AsyncGenerator


class AgentWrapper:
    def __init__(
        self,
        model_wrapper: ModelWrapper,  # 模型包装
        deps_type: type,  # 依赖
        system_prompt: str,  # 系统提示词
    ):
        self.agent = Agent[deps_type, str](
            model_wrapper.model,
            deps_type=deps_type,
            system_prompt=system_prompt,
        )

    async def run(
        self, user_prompt: str, output_type: type[BaseModel]
    ) -> AsyncGenerator:
        async with self.agent.iter(
            user_prompt=user_prompt, output_type=output_type
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
