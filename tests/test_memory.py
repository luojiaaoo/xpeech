import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from xpeech.agent.memory import ConsolidationResult, MemoryConsolidator, MemoryStore
from xpeech.agent.prompt import system as system_prompt_module
from xpeech.agent.tool_executor import ToolExecutionResult, ToolExecutor
from xpeech.provider.schema import LLMResponse, ToolCallChunk, ToolCallRequest


def make_response(
    content: str | None = None,
    tool_calls: list[ToolCallRequest] | None = None,
    mapping_tool_call_funcs=None,
) -> LLMResponse:
    async def chunks():
        if content is not None:
            yield "content", content
        for index, tool_call in enumerate(tool_calls or []):
            yield "tool_calls", ToolCallChunk(
                index=index,
                id=tool_call.id,
                name=tool_call.name,
                arguments=json.dumps(tool_call.arguments, ensure_ascii=False),
            )

    response = LLMResponse(
        iter_mix_chunks=chunks(),
        mapping_tool_call_funcs=mapping_tool_call_funcs,
    )
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", [None, "用户偏好简洁回答"])
async def test_system_prompt_reads_memory(tmp_path, monkeypatch, memory):
    store = MemoryStore(tmp_path)
    if memory is not None:
        await store.write_long_term(memory)
    monkeypatch.setattr(
        system_prompt_module.SkillsLoader, "build_skills_prompt", AsyncMock(return_value=""),
    )

    prompt = await asyncio.wait_for(system_prompt_module.build_system_prompt(tmp_path), 5)

    assert prompt["role"] == "system"
    if memory is not None:
        assert f"## Long-term Memory\n{memory}" in prompt["content"]
    else:
        assert "## Long-term Memory" not in prompt["content"]


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

        chat_kwargs = None

        async def chat(**kwargs):
            nonlocal chat_kwargs
            chat_kwargs = kwargs
            return make_response(
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
        assert chat_kwargs is not None
        assert chat_kwargs["parameters"].max_tokens == 100

    @pytest.mark.asyncio
    async def test_response_without_tool_call_is_skipped(self, tmp_path: Path):
        async def chat(**_kwargs):
            return make_response(content="nothing to save")

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shared_consolidator", [False, True])
    async def test_parallel_sessions_merge_from_latest_memory(self, tmp_path, shared_consolidator):
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = []
        executor = ToolExecutor(workspace=tmp_path, max_result_chars=10_000)

        async def chat(**kwargs):
            context = kwargs["messages"][0]["content"]
            fact = kwargs["messages"][1]["content"]
            calls.append(fact)
            if fact == "first fact":
                first_started.set()
                await release_first.wait()
            if fact == "second fact":
                assert "first fact" in context
            save_memory = kwargs["tools"][0]
            call = ToolCallRequest(
                id=fact,
                name="save_memory",
                arguments={"history_entry": fact, "memory_update": context + "\n" + fact},
            )
            return make_response(
                tool_calls=[call], mapping_tool_call_funcs={"save_memory": save_memory},
            )

        def make_consolidator(workspace):
            return MemoryConsolidator(
                store=MemoryStore(workspace), chat=chat,
                execute_tools=executor.execute, summary_tokens=100,
            )

        consolidator = make_consolidator(tmp_path)
        first = asyncio.create_task(consolidator.consolidate([
            {"role": "user", "content": "first fact"},
        ]))
        second = None
        try:
            await asyncio.wait_for(first_started.wait(), 5)
            next_consolidator = consolidator if shared_consolidator else make_consolidator(tmp_path)
            second = asyncio.create_task(next_consolidator.consolidate([
                {"role": "user", "content": "second fact"},
            ]))
            await asyncio.sleep(0)
            assert calls == ["first fact"]
            # 其他用户无需等待当前用户完成记忆整理。
            other = await asyncio.wait_for(make_consolidator(tmp_path / "other-user").consolidate([
                {"role": "user", "content": "independent fact"},
            ]), 5)
            assert other.status == "saved"
        finally:
            release_first.set()
            assert (await first).status == "saved"
            if second is not None:
                assert (await second).status == "saved"

        store = MemoryStore(tmp_path)
        memory = await store.get_memory_context()
        assert "first fact" in memory
        assert "second fact" in memory
        assert (tmp_path / "memory/HISTORY.md").read_text(encoding="utf-8") == (
            "first fact\n\nsecond fact\n\n"
        )

    @pytest.mark.asyncio
    async def test_cancellation_releases_consolidation_lock(self, tmp_path):
        started = asyncio.Event()

        async def chat(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        async def execute_tools(*_args):
            pytest.fail("no tool execution expected")

        consolidator = MemoryConsolidator(
            store=MemoryStore(tmp_path), chat=chat, execute_tools=execute_tools, summary_tokens=100,
        )
        task = asyncio.create_task(consolidator.consolidate([{"role": "user", "content": "fact"}]))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        store = MemoryStore(tmp_path)
        await asyncio.wait_for(store.write_long_term("next session"), 5)
        assert await store.get_memory_context() == "## Long-term Memory\nnext session"
