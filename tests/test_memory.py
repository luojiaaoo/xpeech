from pathlib import Path

import pytest
from pydantic import BaseModel

from xpeech.agent.memory import ConsolidationResult, MemoryConsolidator, MemoryStore
from xpeech.agent.tool_executor import ToolExecutionResult
from xpeech.provider.schema import LLMResponse, ToolCallRequest


class TestMemoryConsolidator:
    @pytest.mark.asyncio
    async def test_empty_conversation_is_skipped_without_calling_model(self, tmp_path: Path):
        async def chat(**_kwargs):
            pytest.fail("chat should not be called for an empty conversation")

        async def execute_tools(*_args, **_kwargs):
            pytest.fail("tools should not be executed for an empty conversation")

        consolidator = MemoryConsolidator(
            store=MemoryStore(tmp_path),
            chat=chat,
            execute_tools=execute_tools,
            summary_tokens=100,
        )

        result = await consolidator.consolidate([{"role": "system", "content": "system"}])

        assert result == ConsolidationResult(status="skipped", message="当前上下文为空，无需记忆")

    @pytest.mark.asyncio
    async def test_success_requires_save_memory_tool_to_succeed(self, tmp_path: Path):
        tool_call = ToolCallRequest(
            id="call-1",
            name="save_memory",
            arguments={"history_entry": "entry", "memory_update": "memory"},
        )

        def save_memory(_args: type[BaseModel] | None) -> str:
            return "saved"

        async def chat(**_kwargs):
            return LLMResponse(
                content=None,
                tool_calls=[tool_call],
                mapping_tool_call_funcs={"save_memory": save_memory},
            )

        async def execute_tools(tool_calls, mapping_tool_call_funcs, loop_count=None):
            assert tool_calls == [tool_call]
            assert "save_memory" in mapping_tool_call_funcs
            assert loop_count is None
            return [
                ToolExecutionResult(
                    call=tool_call,
                    value="saved",
                    succeeded=True,
                    duration_seconds=0.1,
                )
            ]

        consolidator = MemoryConsolidator(
            store=MemoryStore(tmp_path),
            chat=chat,
            execute_tools=execute_tools,
            summary_tokens=100,
        )

        result = await consolidator.consolidate([{"role": "user", "content": "remember this"}])

        assert result == ConsolidationResult(status="saved", message="已记忆本次会话关键内容")

    @pytest.mark.asyncio
    async def test_response_without_tool_call_is_skipped(self, tmp_path: Path):
        async def chat(**_kwargs):
            return LLMResponse(content="nothing to save")

        async def execute_tools(*_args, **_kwargs):
            pytest.fail("tools should not be executed without tool calls")

        consolidator = MemoryConsolidator(
            store=MemoryStore(tmp_path),
            chat=chat,
            execute_tools=execute_tools,
            summary_tokens=100,
        )

        result = await consolidator.consolidate([{"role": "user", "content": "hello"}])

        assert result == ConsolidationResult(status="skipped", message="未发现需要记忆的内容")
