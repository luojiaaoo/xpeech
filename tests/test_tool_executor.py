from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.provider.schema import ToolCallRequest


class EchoArgs(BaseModel):
    text: Annotated[str, Field(description="Text to echo")]


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_executes_tools_and_preserves_request_order(self):
        async def echo(args: EchoArgs) -> str:
            return args.text

        calls = [
            ToolCallRequest(id="2", name="echo", arguments={"text": "second"}),
            ToolCallRequest(id="1", name="echo", arguments={"text": "first"}),
        ]

        results = await ToolExecutor().execute(calls, {"echo": echo}, loop_count=3)

        assert [result.call.id for result in results] == ["2", "1"]
        assert [result.value for result in results] == ["second", "first"]
        assert all(result.succeeded for result in results)
        assert all(result.error is None for result in results)

    @pytest.mark.asyncio
    async def test_returns_failure_for_unregistered_tool(self):
        call = ToolCallRequest(id="1", name="missing", arguments={})

        [result] = await ToolExecutor().execute([call], {})

        assert not result.succeeded
        assert result.value == "ValueError: Tool is not registered for this request: missing"
        assert result.error == result.value
