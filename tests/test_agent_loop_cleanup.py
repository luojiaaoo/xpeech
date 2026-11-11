"""Interrupted Agent turns must be durable before their owner task exits."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi import HTTPException

from xpeech.agent import loop as loop_module
from xpeech.agent.history import YamlHistoryRepository
from xpeech.agent.loop import AgentLoop
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.provider.schema import LLMResponse


def response(*chunks):
    async def stream():
        for chunk in chunks:
            yield chunk

    return LLMResponse(iter_mix_chunks=stream())


@pytest.fixture
def turn(monkeypatch, tmp_path):
    monkeypatch.setattr(loop_module, "get_request_id", lambda: "test-request")
    monkeypatch.setattr(loop_module, "build_system_prompt", AsyncMock(return_value={"role": "system", "content": "sys"}))
    monkeypatch.setattr(loop_module, "build_user_prompt", AsyncMock(return_value={"role": "user", "content": "hi"}))
    agent = AgentLoop.__new__(AgentLoop)
    agent.workspace = tmp_path
    agent.channel = "test"
    agent.user_organization = None
    agent.history = YamlHistoryRepository(tmp_path / "history")
    agent.history.save = AsyncMock(wraps=agent.history.save)
    agent.compressor = SimpleNamespace(should_compress=AsyncMock(return_value=False))
    agent.chat = AsyncMock(return_value=response(("content", "answer")))
    agent.records = SimpleNamespace(append=AsyncMock())
    agent.tools = []
    agent.max_iterations = 3
    agent.max_accept_token = 1000
    agent._input_tokens = agent._output_tokens = agent._model_call_count = 0
    agent._count_context_tokens = AsyncMock(return_value=10)
    message = InboundMessage(
        user_id="user", session_id="018f0000-0000-7000-8000-000000000001", sender_name="Alice",
        channel="test", content=[InputText(text="hi")], session_metadata={}, files=[],
    )
    return agent, message


async def saved_turn(agent, message):
    agent.history.save.assert_awaited_once()
    return await agent.history.load(message.user_id, message.session_id, channel=message.channel)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["compression", "provider", "http"])
async def test_errors_persist_user_input_once(turn, stage):
    agent, message = turn
    if stage == "compression":
        agent.compressor.should_compress.side_effect = ValueError("compression failed")
    elif stage == "provider":
        agent.chat.side_effect = ValueError("provider failed")
    else:
        agent.chat.side_effect = HTTPException(429, "limited")
    with pytest.raises(HTTPException if stage != "compression" else ValueError):
        _ = [event async for event in agent.run(message)]
    saved = await saved_turn(agent, message)
    assert saved[0] == {"role": "user", "content": "hi"}
    agent.records.append.assert_not_awaited()


@pytest.mark.asyncio
async def test_closing_at_fallback_yield_saves_the_assistant_message(turn):
    agent, message = turn
    agent.chat.return_value = response()
    events = agent.run(message)
    fallback = await anext(events)
    assert fallback["event"] == "assistant"
    await events.aclose()
    saved = await saved_turn(agent, message)
    assert saved[-1] == {"role": "assistant", "content": fallback["context"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["close", "cancel", "anyio"])
async def test_interruption_persists_existing_context(turn, exit_kind):
    agent, message = turn
    reading = asyncio.Event()

    async def chunks():
        yield "content", "partial"
        reading.set()
        await anyio.sleep_forever()

    agent.chat.return_value = LLMResponse(iter_mix_chunks=chunks())
    if exit_kind == "close":
        events = agent.run(message)
        await anext(events)
        await events.aclose()
    elif exit_kind == "anyio":
        with anyio.CancelScope() as scope:
            async for _event in agent.run(message):
                scope.cancel()
    else:
        async def consume():
            return [event async for event in agent.run(message)]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(reading.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert await saved_turn(agent, message) == [{"role": "user", "content": "hi"}]
    agent.records.append.assert_not_awaited()
