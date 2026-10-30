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
@pytest.mark.parametrize("json_output", [False, True])
async def test_not_interactive_run_saves_separate_yaml_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_output):
    async def build_test_system_prompt(workspace: Path, *, channel_tags):
        assert workspace == tmp_path
        return {"role": "system", "content": "system"}

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)
    monkeypatch.setattr(loop_module, "token_counter", AsyncMock(return_value=10))

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.workspace = tmp_path
    agent_loop.channel_tags = []
    agent_loop.history = SimpleNamespace(
        load=AsyncMock(return_value=[{"role": "user", "content": "old context"}]),
        save=AsyncMock(),
        delete=AsyncMock(),
    )
    agent_loop.memory_consolidator = SimpleNamespace(consolidate=AsyncMock())
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
        user_id="session-1",
        session_id="chat-1",
        sender_name="Scheduler",
        channel="feishu",
        session_metadata={"source": "scheduled_task"},
        content=[InputText(text="/new Run scheduled prompt")],
        timestamp="2026-09-06T10:00:00",
        files=[],
        json_output=json_output,
    )

    events = [event async for event in agent_loop.run(message, not_interactive=True)]

    agent_loop.history.load.assert_not_awaited()
    agent_loop.history.save.assert_awaited_once()
    assert agent_loop.history.save.await_args.kwargs["not_interactive"] is True
    saved_user_id, saved_session_id, saved_messages = agent_loop.history.save.await_args.args
    assert saved_user_id == message.user_id
    assert saved_session_id == message.session_id
    assert saved_messages[-1] == {"role": "assistant", "content": "scheduled result"}
    agent_loop.history.delete.assert_not_awaited()
    agent_loop.memory_consolidator.consolidate.assert_not_awaited()
    messages = agent_loop.chat.await_args.kwargs["messages"]
    assert all(item.get("content") != "old context" for item in messages)
    assert "/new Run scheduled prompt" in [
        part["text"]
        for part in messages[1]["content"]
        if part.get("type") == "text"
    ]
    assert {"event": "assistant", "context": "scheduled result"} in events
    assert "remove_interactive_tool" not in agent_loop.chat.await_args.kwargs
    assert agent_loop.chat.await_args.kwargs["json_output"] is json_output
    assert messages[0] == {"role": "system", "content": "system"}
    assert all("valid JSON object" not in part["text"] for part in messages[1]["content"])


@pytest.mark.asyncio
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_provider_failure_persists_messages_and_raises_http_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    not_interactive: bool,
):
    async def build_test_system_prompt(workspace: Path, *, channel_tags):
        assert workspace == tmp_path
        return {"role": "system", "content": "system"}

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.workspace = tmp_path
    agent_loop.channel_tags = []
    agent_loop.history = SimpleNamespace(
        load=AsyncMock(return_value=[]),
        save=AsyncMock(),
        delete=AsyncMock(),
    )
    agent_loop.memory_consolidator = SimpleNamespace(consolidate=AsyncMock())
    agent_loop.compressor = SimpleNamespace(
        should_compress=AsyncMock(return_value=False),
    )
    agent_loop.chat = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    agent_loop.records = SimpleNamespace(append=AsyncMock())
    agent_loop.tools = []
    agent_loop.max_iterations = 2
    agent_loop.max_accept_token = 100_000
    agent_loop._input_tokens = 0
    agent_loop._output_tokens = 0
    agent_loop._model_call_count = 0
    message = InboundMessage(
        user_id="session-1",
        session_id="chat-1",
        sender_name="Alice",
        channel="test",
        session_metadata={},
        content=[InputText(text="hello")],
        timestamp="2026-09-06T10:00:00",
        files=[],
    )

    with pytest.raises(loop_module.HTTPException) as exc_info:
        _ = [event async for event in agent_loop.run(message, not_interactive=not_interactive)]

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "provider unavailable"
    save = agent_loop.history.save
    save.assert_awaited_once()
    assert save.await_args.kwargs["not_interactive"] is not_interactive
    saved_user_id, saved_identifier, saved_messages = save.await_args.args
    assert saved_user_id == "session-1"
    assert saved_identifier == message.session_id
    assert saved_messages[-1] == {
        "role": "user",
        "content": (
            "The previous model call failed with this error: provider unavailable. "
            "Please retry the request and adjust your approach if needed."
        ),
    }
    assert any(
        part.get("text") == "hello"
        for part in saved_messages[-2]["content"]
        if isinstance(part, dict)
    )
