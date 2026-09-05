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
@pytest.mark.parametrize(
    ("command", "command_result", "consolidates_memory"),
    [
        ("/new continue this task", "新会话, memory saved", True),
        ("/clear continue this task", "上下文已清空", False),
    ],
)
async def test_context_commands_continue_with_trailing_prompt(
    command: str,
    command_result: str,
    consolidates_memory: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    async def build_test_system_prompt(workspace: Path):
        assert workspace == tmp_path
        return {"role": "system", "content": "system"}

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)
    monkeypatch.setattr(loop_module, "token_counter", AsyncMock(return_value=10))

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.workspace = tmp_path
    load_results = (
        [[{"role": "user", "content": "old context"}], []]
        if consolidates_memory
        else [[]]
    )
    agent_loop.history = SimpleNamespace(
        load=AsyncMock(side_effect=load_results),
        save=AsyncMock(),
        delete=AsyncMock(),
    )
    agent_loop.memory_consolidator = SimpleNamespace(
        consolidate=AsyncMock(return_value=SimpleNamespace(message="memory saved")),
    )
    agent_loop.compressor = SimpleNamespace(
        should_compress=AsyncMock(return_value=False),
    )
    agent_loop.chat = AsyncMock(return_value=make_response("continued result"))
    agent_loop.records = SimpleNamespace(append=AsyncMock())
    agent_loop.tools = []
    agent_loop.max_iterations = 2
    agent_loop.max_accept_token = 100_000
    agent_loop._input_tokens = 0
    agent_loop._output_tokens = 0
    agent_loop._model_call_count = 0
    message = InboundMessage(
        session_id="session-1",
        sender_name="Alice",
        session_metadata={},
        content=[InputText(text=command)],
        timestamp="2026-09-06T10:00:00",
        files=[],
    )

    events = [event async for event in agent_loop.run(message)]

    assert events[0] == {"event": "command", "context": command_result}
    assert {"event": "assistant", "context": "continued result"} in events
    agent_loop.history.delete.assert_awaited_once_with("session-1")
    if consolidates_memory:
        assert agent_loop.history.load.await_count == 2
        agent_loop.memory_consolidator.consolidate.assert_awaited_once_with(
            [{"role": "user", "content": "old context"}],
        )
    else:
        agent_loop.history.load.assert_awaited_once_with("session-1")
        agent_loop.memory_consolidator.consolidate.assert_not_awaited()

    model_messages = agent_loop.chat.await_args.kwargs["messages"]
    assert len(model_messages) == 3
    assert model_messages[0] == {"role": "system", "content": "system"}
    prompt_texts = [part["text"] for part in model_messages[1]["content"]]
    assert "continue this task" in prompt_texts
    assert command not in prompt_texts
    assert model_messages[2] == {"role": "assistant", "content": "continued result"}
    assert all(message.get("content") != "old context" for message in model_messages)
