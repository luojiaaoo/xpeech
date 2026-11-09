import asyncio
import json
from types import SimpleNamespace

import pytest

from xpeech.agent.loop import AgentLoop
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools.interactions import SendFileRequest
from xpeech.provider.schema import ToolCallRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("fast_name", ["fast_tool", "missing", "send_file"])
async def test_tool_results_are_sent_individually_as_tools_finish(tmp_path, fast_name):
    release_slow = asyncio.Event()

    async def slow_tool() -> str:
        await release_slow.wait()
        return "slow result"

    async def fast_tool():
        return SendFileRequest(source="fast result") if fast_name == "send_file" else "fast result"

    calls = [
        ToolCallRequest(id="slow", name="slow_tool", arguments={}),
        ToolCallRequest(id="fast", name=fast_name, arguments={}),
    ]
    mapping = {"slow_tool": slow_tool}
    if fast_name != "missing":
        mapping[fast_name] = fast_tool
    response = SimpleNamespace(tool_calls=calls, content="working")
    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.tool_executor = ToolExecutor(
        workspace=tmp_path, max_result_chars=10_000, mapping_tool_call_funcs=mapping,
    )
    history = []

    events = []
    results = []
    async with asyncio.timeout(1):
        async for event in agent_loop.tool_call(response, history, 0, "user", "session", "Alice"):
            events.append(event)
            if event["event"] == "tool_call_result":
                result = json.loads(event["context"])
                assert set(result) == {"id", "name", "result", "duration_seconds"}
                results.append(result)
                if result["id"] == "fast":
                    assert not release_slow.is_set()
                    assert [message["tool_call_id"] for message in history if message["role"] == "tool"] == ["fast"]
                    release_slow.set()

    assert events[0]["event"] == "tool_call"
    if fast_name == "send_file":
        assert events[1] == {"event": "send_file", "context": "fast result"}
    first_result, second_result = results
    assert (first_result["id"], first_result["name"]) == ("fast", fast_name)
    assert first_result["duration_seconds"] >= 0
    if fast_name == "missing":
        assert "not registered" in first_result["result"]
    else:
        assert "fast result" in first_result["result"]
    assert (second_result["id"], second_result["name"], second_result["result"]) == (
        "slow", "slow_tool", "slow result"
    )

    assert [message["tool_call_id"] for message in history if message["role"] == "tool"] == ["fast", "slow"]


@pytest.mark.asyncio
async def test_cancelling_tool_events_cancels_pending_tools(tmp_path):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    result_received = asyncio.Event()

    async def slow_tool() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def fast_tool() -> str:
        await started.wait()
        return "ready"

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.tool_executor = ToolExecutor(
        workspace=tmp_path, max_result_chars=10_000,
        mapping_tool_call_funcs={"slow_tool": slow_tool, "fast_tool": fast_tool},
    )
    response = SimpleNamespace(
        tool_calls=[
            ToolCallRequest(id="slow", name="slow_tool", arguments={}),
            ToolCallRequest(id="fast", name="fast_tool", arguments={}),
        ],
        content="",
    )

    async def consume():
        async for event in agent_loop.tool_call(response, [], 0, "user", "session", "Alice"):
            if event["event"] == "tool_call_result":
                result_received.set()

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(result_received.wait(), timeout=1)
    finally:
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    assert cancelled.is_set()
