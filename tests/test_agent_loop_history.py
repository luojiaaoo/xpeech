from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import xpeech.agent.loop as loop_module
from xpeech.agent.loop import AgentLoop
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.provider.schema import LLMResponse


def make_response(content: str) -> LLMResponse:
    async def chunks():
        yield "content", content

    return LLMResponse(iter_mix_chunks=chunks())


@pytest.mark.asyncio
async def test_run_can_skip_yaml_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def build_test_system_prompt(workspace: Path):
        assert workspace == tmp_path
        return {"role": "system", "content": "system"}

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)
    monkeypatch.setattr(loop_module, "token_counter", AsyncMock(return_value=10))

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.workspace = tmp_path
    agent_loop.history = SimpleNamespace(
        load=AsyncMock(return_value=[{"role": "user", "content": "old context"}]),
        save=AsyncMock(),
    )
    agent_loop.compressor = SimpleNamespace(
        should_compress=AsyncMock(return_value=False),
    )
    agent_loop.chat = AsyncMock(return_value=make_response("scheduled result"))
    agent_loop.records = SimpleNamespace(append=AsyncMock())
    agent_loop.tools = []
    agent_loop.max_iterations = 2
    agent_loop.max_accept_token = 100_000
    agent_loop._input_tokens = 0
    agent_loop._output_tokens = 0
    agent_loop._model_call_count = 0
    message = InboundMessage(
        session_id="session-1",
        sender_name="Scheduler",
        session_metadata={"source": "scheduled_task"},
        content=[InputText(text="Run scheduled prompt")],
        timestamp="2026-09-06T10:00:00",
        files=[],
    )

    events = [event async for event in agent_loop.run(message, use_history=False)]

    agent_loop.history.load.assert_not_awaited()
    agent_loop.history.save.assert_not_awaited()
    messages = agent_loop.chat.await_args.kwargs["messages"]
    assert all(item.get("content") != "old context" for item in messages)
    assert {"event": "assistant", "context": "scheduled result"} in events
