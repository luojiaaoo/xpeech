"""Interrupted Agent turns must be durable before their owner task exits."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from fastapi import FastAPI, HTTPException

from xpeech.agent import execute
from xpeech.agent import loop as loop_module
from xpeech.agent.history import YamlHistoryRepository
from xpeech.agent.loop import AgentLoop
from xpeech.agent.model_anomaly import ModelAnomalyGuard
from xpeech.agent.prompt.helper import build_user_prompt
from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.auth import require_user_jwt
from xpeech.agent.server.routes.chat import router
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.agent.server.session_guard import SessionChatGuard, SessionInbox
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools.execution_request import EXECUTION_DESCRIPTION
from xpeech.agent.tools.interactions import ExecutionRequest, QuestionRequest
from xpeech.config.settings import settings
from xpeech.provider.schema import LLMResponse, ToolCallChunk


def response(*chunks):
    async def stream():
        for chunk in chunks:
            yield chunk

    return LLMResponse(iter_mix_chunks=stream())


@pytest.fixture
def turn(monkeypatch, tmp_path):
    monkeypatch.setattr(loop_module, "build_context_messages", AsyncMock(return_value=[]))
    monkeypatch.setattr(loop_module, "get_request_id", lambda: "test-request")
    monkeypatch.setattr(loop_module, "build_system_prompt", AsyncMock(return_value={"role": "system", "content": "sys"}))
    monkeypatch.setattr(loop_module, "build_user_prompt", AsyncMock(return_value={"role": "user", "content": "hi"}))
    agent = AgentLoop.__new__(AgentLoop)
    agent.provider = SimpleNamespace(
        default_context_token=200_000, default_model="test",
        chat_count_tool_tokens=AsyncMock(return_value={"system_tools": 0, "mcp_tools": 0}),
    )
    agent.workspace = tmp_path
    agent.channel = "test"
    agent.user_organization = None
    agent.history = YamlHistoryRepository(tmp_path / "history")
    agent.history.save = AsyncMock(wraps=agent.history.save)
    agent.compressor = SimpleNamespace(should_compress=AsyncMock(return_value=False))
    agent.chat = AsyncMock(return_value=response(("content", "answer")))
    agent.records = SimpleNamespace(append=AsyncMock())
    agent.tools = []
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
@pytest.mark.parametrize("first_call_uses_tool", [False, True])
async def test_chat_inserts_each_queued_message_after_first_tool_free_iteration(
    turn, monkeypatch, first_call_uses_tool,
):
    agent, message = turn
    app = FastAPI()
    app.include_router(router)
    auth = SimpleNamespace(user_id=message.user_id, channel=message.channel)
    app.dependency_overrides[require_user_jwt] = lambda: auth
    monkeypatch.setattr(settings.path, "workspace_base_path", agent.workspace)
    runner = AgentRunner(message.user_id, config=settings)
    agent.workspace = runner.workspace
    runner._agent_loop = agent
    create = AsyncMock(return_value=runner)
    monkeypatch.setattr(AgentRunner, "create", create)
    monkeypatch.setattr(loop_module, "build_user_prompt", build_user_prompt)
    waiting, resume = asyncio.Event(), asyncio.Event()
    calls = []

    async def first_response():
        waiting.set()
        await resume.wait()
        if first_call_uses_tool:
            yield "tool_calls", ToolCallChunk(index=0, id="call-1", name="lookup", arguments="{}")
        else:
            yield "content", "first answer"

    async def lookup():
        return "tool result"

    agent.tool_executor = ToolExecutor(
        workspace=agent.workspace, max_result_chars=1000, mapping_tool_call_funcs={"lookup": lookup},
    )

    async def model_call(*, messages, tools, json_output):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return LLMResponse(iter_mix_chunks=first_response())
        if first_call_uses_tool and len(calls) == 2:
            return response(("content", "first answer"))
        return response(("content", "final answer"))

    agent.chat = model_call

    async def consume():
        async with execute.execute_chat_stream(
            message.user_id, channel=message.channel, session_id=message.session_id, sender_name=message.sender_name,
            content=message.content, session_metadata={}, files=[], json_output=False,
        ) as events:
            return [event async for event in events]

    owner = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(waiting.wait(), timeout=2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for n in (1, 2):
                queued = await asyncio.wait_for(client.post(
                    "/chat",
                    headers={"x-session-id": message.session_id, "x-sender-name": message.sender_name},
                    data={"content": json.dumps([{"text": f"follow-up {n}"}])},
                    files=[("files", ("note.txt", b"attachment"))] if n == 2 else None,
                ), timeout=2)
                assert queued.status_code == 200
                [ack] = [json.loads(line[6:]) for line in queued.text.splitlines() if line.startswith("data: ")]
                assert ack["event"] == "command"
        resume.set()
        events = await asyncio.wait_for(owner, timeout=2)
    finally:
        if not owner.done():
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    assert create.await_count == 3
    assert not any(event["event"] == "user" for event in events)
    if first_call_uses_tool:
        assert [msg["role"] for msg in calls[1]] == ["system", "user", "assistant", "tool"]
    assert len(calls) == (3 if first_call_uses_tool else 2)
    reply, first, second = calls[-1][-3:]
    assert reply == {"role": "assistant", "content": "first answer"}
    assert [first["role"], second["role"]] == ["user", "user"]
    assert [first["content"][1]["text"], second["content"][1]["text"]] == ["follow-up 1", "follow-up 2"]
    assert "sender_name: Alice" in second["content"][0]["text"]
    assert "note.txt" in second["content"][2]["text"]
    assert (await saved_turn(agent, message))[-4:-1] == [reply, first, second]
    assert agent.records.append.await_args.kwargs["messages"][-4:-1] == [reply, first, second]


@pytest.mark.asyncio
@pytest.mark.parametrize("json_output", [False, True])
async def test_queued_messages_preserve_tool_context_and_json_output(turn, monkeypatch, json_output):
    agent, message = turn
    monkeypatch.setattr(loop_module, "build_user_prompt", build_user_prompt)
    message.session_metadata = {"open_id": "alice-open-id", "initial_only": "value"}
    message.json_output = json_output
    inbox = SessionInbox(message)
    followups = [
        message.model_copy(update={
            "content": [InputText(text=f"Follow-up {index}")],
            "json_output": not json_output,
        })
        for index in (1, 2)
    ]
    tool_contexts = []

    async def capture(sender_name: str | None, session_metadata: dict[str, str] | None) -> str:
        tool_contexts.append((sender_name, deepcopy(session_metadata)))
        return "captured"

    agent.tool_executor = ToolExecutor(
        workspace=agent.workspace, max_result_chars=1000, mapping_tool_call_funcs={"capture": capture},
    )
    calls = 0

    async def model_call(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["json_output"] is json_output
        if calls == 1:
            for incoming in followups:
                assert inbox.enqueue(incoming)
        if calls in (1, 3, 4):
            return response(("tool_calls", ToolCallChunk(
                index=0, id=f"call-{calls}", name="capture", arguments="{}",
            )))
        return response(("content", "first answer" if calls == 2 else "final answer"))

    agent.chat = model_call
    events = [event async for event in agent.run(message, inbox=inbox)]

    assert calls == 5
    assert tool_contexts == [("Alice", message.session_metadata)] * 3
    assert not any(event["event"] == "user" for event in events)
    assert agent.records.append.await_args.args[0].sender_name == "Alice"
    assert message.session_metadata == {"open_id": "alice-open-id", "initial_only": "value"}


@pytest.mark.asyncio
@pytest.mark.parametrize("inbox_mode", ["none", "empty", "pending"])
async def test_tool_chain_continues_until_no_tools_and_no_pending_messages(turn, inbox_mode):
    agent, message = turn
    inbox = None if inbox_mode == "none" else SessionInbox(message)
    calls = 0

    async def lookup():
        return "tool result"

    agent.tool_executor = ToolExecutor(
        workspace=agent.workspace, max_result_chars=1000, mapping_tool_call_funcs={"lookup": lookup},
    )

    async def model_call(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1 and inbox_mode == "pending":
            assert inbox.enqueue(message)
        # Continue past the previous default of 40, including without an inbox.
        if calls <= 45 or calls == 47:
            return response(("tool_calls", ToolCallChunk(
                index=0, id=f"call-{calls}", name="lookup", arguments="{}",
            )))
        return response(("content", f"answer {calls}"))

    agent.chat = model_call
    events = [event async for event in agent.run(message, inbox=inbox)]

    expected_calls = 48 if inbox_mode == "pending" else 46
    assert calls == expected_calls
    assert (await saved_turn(agent, message))[-1] == {
        "role": "assistant", "content": f"answer {expected_calls}",
    }
    answers = [event["context"] for event in events if event["event"] == "assistant"]
    assert answers == (["answer 46", "answer 48"] if inbox_mode == "pending" else ["answer 46"])
    if inbox is not None:
        assert not inbox.enqueue(message)
        assert inbox.drain() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("batches, expected", [
    ([[("lookup", {})]] * 2, []),
    ([[("lookup", {})]] * 5, [("lookup", 3)]),
    ([[("lookup", {"x": 1, "y": 2})], [("lookup", {"y": 2, "x": 1})],
      [("lookup", {"x": 1, "y": 2})]], [("lookup", 3)]),
    ([[("lookup", {})]] * 2 + [[("other", {})]] + [[("lookup", {})]] * 3, [("lookup", 3)]),
    ([[("lookup", {})]] * 2 + [[("lookup", {"x": 1})]] * 3, [("lookup", 3)]),
    ([[("lookup", {})] * 3], [("lookup", 3)]),
    ([[("lookup", {})] * 2] * 2, [("lookup", 3)]),
    ([[("lookup", {})]] * 4 + [[("other", {})]] + [[("lookup", {})]] * 4,
     [("lookup", 3), ("lookup", 3)]),
])
async def test_repeated_tool_calls_inject_reminders_without_splitting_results(turn, batches, expected):
    agent, message = turn
    calls = []

    async def lookup():
        return "result"

    agent.tool_executor = ToolExecutor(
        workspace=agent.workspace, max_result_chars=1000,
        mapping_tool_call_funcs={"lookup": lookup, "other": lookup},
    )

    async def model_call(**kwargs):
        calls.append(deepcopy(kwargs["messages"]))
        index = len(calls) - 1
        if index == len(batches):
            return response(("content", "done"))
        return response(*[
            ("tool_calls", ToolCallChunk(
                index=offset, id=f"call-{index}-{offset}", name=name, arguments=json.dumps(arguments),
            ))
            for offset, (name, arguments) in enumerate(batches[index])
        ])

    agent.chat = model_call
    _ = [event async for event in agent.run(message)]
    expected_messages = [
        {
            "role": "user",
            "content": ModelAnomalyGuard.REPEATED_TOOL_CALL_PROMPT.format(tool_name=name, observed_count=count),
        }
        for name, count in expected
    ]
    await saved_turn(agent, message)
    saved = agent.history.save.await_args.args[2]

    def reminders(messages):
        return [
            msg for msg in messages
            if msg["role"] == "user" and str(msg["content"]).startswith("You have called ")
        ]

    assert [msg for context in calls for msg in reminders(context)] == expected_messages
    assert reminders(saved) == []
    assert reminders(agent.records.append.await_args.kwargs["messages"]) == []
    # Reminders must not split tool calls from their results, including parallel batches.
    for context in calls[1:]:
        outstanding = set()
        completed_tools = 0
        for msg in context:
            if msg["role"] == "assistant":
                outstanding.update(call["id"] for call in msg.get("tool_calls", []))
            elif msg["role"] == "tool":
                outstanding.remove(msg["tool_call_id"])
                completed_tools += 1
            elif msg["role"] == "user":
                assert not outstanding
                if str(msg["content"]).startswith("You have called "):
                    assert completed_tools >= 3


@pytest.mark.asyncio
async def test_anomaly_reminder_is_used_once_after_compression_and_excluded_from_memory(turn):
    agent, message = turn
    agent.compressor.should_compress.side_effect = [False, False, False, True, False]
    agent.compressor.compress = AsyncMock(return_value=[
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "compressed history"},
    ])
    agent.memory_consolidator = SimpleNamespace(consolidate=AsyncMock())
    calls = []

    async def lookup():
        return "result"

    agent.tool_executor = ToolExecutor(
        workspace=agent.workspace, max_result_chars=1000, mapping_tool_call_funcs={"lookup": lookup},
    )

    async def model_call(**kwargs):
        calls.append(deepcopy(kwargs["messages"]))
        if len(calls) == 5:
            return response(("content", "done"))
        return response(("tool_calls", ToolCallChunk(
            index=0, id=f"call-{len(calls)}", name="lookup", arguments="{}",
        )))

    agent.chat = model_call
    _ = [event async for event in agent.run(message)]
    reminder = {
        "role": "user",
        "content": ModelAnomalyGuard.REPEATED_TOOL_CALL_PROMPT.format(tool_name="lookup", observed_count=3),
    }
    assert [reminder in messages for messages in calls] == [False, False, False, True, False]
    assert calls[3] == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "compressed history"},
        reminder,
    ]
    agent.compressor.compress.assert_awaited_once()
    agent.memory_consolidator.consolidate.assert_awaited_once()
    assert reminder not in agent.compressor.compress.await_args.args[0]
    assert reminder not in agent.memory_consolidator.consolidate.await_args.args[0]
    assert reminder not in agent.history.save.await_args.args[2]
    assert reminder not in agent.records.append.await_args.kwargs["messages"]


@pytest.mark.asyncio
async def test_inbox_message_resets_repeated_tool_call_count(turn):
    agent, message = turn
    inbox = SessionInbox(message)
    calls = 0
    contexts = []

    async def lookup():
        return "result"

    agent.tool_executor = ToolExecutor(
        workspace=agent.workspace, max_result_chars=1000, mapping_tool_call_funcs={"lookup": lookup},
    )

    async def model_call(**kwargs):
        nonlocal calls
        contexts.append(deepcopy(kwargs["messages"]))
        calls += 1
        if calls == 1:
            assert inbox.enqueue(message)
        if calls in (4, 8):
            return response(("content", "done"))
        return response(("tool_calls", ToolCallChunk(
            index=0, id=f"call-{calls}", name="lookup", arguments="{}",
        )))

    agent.chat = model_call
    _ = [event async for event in agent.run(message, inbox=inbox)]
    await saved_turn(agent, message)
    saved = agent.history.save.await_args.args[2]
    assert not any(str(msg["content"]).startswith("You have called ") for msg in saved)
    reminders = [
        msg for context in contexts for msg in context
        if str(msg["content"]).startswith("You have called ")
    ]
    assert not any(
        str(msg["content"]).startswith("You have called ")
        for msg in agent.records.append.await_args.kwargs["messages"]
    )
    assert calls == 8
    assert reminders == [{
        "role": "user",
        "content": ModelAnomalyGuard.REPEATED_TOOL_CALL_PROMPT.format(tool_name="lookup", observed_count=3),
    }] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("field, value", [
    ("sender_name", "Bob"), ("session_metadata", {"trace": "other"}),
])
async def test_chat_returns_conflict_for_changed_inbox_context(turn, monkeypatch, field, value):
    agent, message = turn
    guard = SessionChatGuard()
    inbox, _ = await guard.submit(message)
    monkeypatch.setattr(execute, "CHAT_GUARD", guard)
    monkeypatch.setattr(AgentRunner, "create", AsyncMock(return_value=SimpleNamespace(workspace=agent.workspace)))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user_jwt] = lambda: SimpleNamespace(
        user_id=message.user_id, channel=message.channel,
    )
    incoming = message.model_copy(update={field: value})
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            result = await asyncio.wait_for(client.post(
                "/chat",
                headers={"x-session-id": message.session_id, "x-sender-name": incoming.sender_name},
                data={
                    "content": json.dumps([{"text": "follow-up"}]),
                    "session_metadata": json.dumps(incoming.session_metadata),
                },
            ), 2)
        assert result.status_code == 409
        assert field in result.json()["detail"]
        assert inbox.drain() == []
    finally:
        await guard.release(message.user_id, message.session_id, inbox)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation", [False, True])
@pytest.mark.parametrize("exit_kind", ["close", "cancel"])
async def test_closing_or_cancelling_question_clears_waiter_immediately(turn, confirmation, exit_kind):
    agent, message = turn
    executions = []

    async def execute():
        executions.append("executed")
        return "done"

    async def ask():
        return ExecutionRequest() if confirmation else QuestionRequest(question='{"fields":[]}')

    ask.execution_function = execute
    agent.mapping_tool_call_funcs = {"ask": ask}
    agent.tool_executor = ToolExecutor(
        workspace=agent.workspace, max_result_chars=1000, mapping_tool_call_funcs=agent.mapping_tool_call_funcs,
    )
    agent.chat.return_value = response(("tool_calls", ToolCallChunk(
        index=0, id="question-1", name="ask", arguments='{"' + EXECUTION_DESCRIPTION + '":"确认执行"}',
    )))
    key = (message.user_id, message.session_id)
    events = agent.run(message)
    question_received = asyncio.Event()
    task = None
    other_key = (message.user_id, "018f0000-0000-7000-8000-000000000002")
    other = loop_module.QuestionEvent(event=asyncio.Event())
    AgentLoop.SESSION_QUESTION_EVENT[other_key] = other
    try:
        if exit_kind == "close":
            async for event in events:
                if event["event"] == "question":
                    break
            assert key in AgentLoop.SESSION_QUESTION_EVENT
            await events.aclose()
        else:
            async def consume():
                async for event in events:
                    if event["event"] == "question":
                        question_received.set()

            task = asyncio.create_task(consume())
            await asyncio.wait_for(question_received.wait(), timeout=1)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert key not in AgentLoop.SESSION_QUESTION_EVENT
        assert AgentLoop.SESSION_QUESTION_EVENT[other_key] is other
        assert executions == []
        assert (await saved_turn(agent, message))[0] == {"role": "user", "content": "hi"}
    finally:
        AgentLoop.SESSION_QUESTION_EVENT.pop(key, None)
        AgentLoop.SESSION_QUESTION_EVENT.pop(other_key, None)
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await events.aclose()
