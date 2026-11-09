import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import xpeech.agent.loop as loop_module
from xpeech.agent.loop import AgentLoop
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import LLMParameters, LLMResponse, ToolCallChunk
from xpeech.utils.helper import token_counter


@pytest.fixture(autouse=True)
def request_context(monkeypatch):
    monkeypatch.setattr(loop_module, "get_request_id", lambda: "00000000-0000-4000-8000-000000000001")


def make_response(content: str) -> LLMResponse:
    async def chunks():
        yield "content", content

    return LLMResponse(iter_mix_chunks=chunks())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_chunks, expected_content",
    [
        ([], "I've completed processing but have no response to give."),
        ([("content", "")], "I've completed processing but have no response to give."),
        ([("content", " \n\t")], "I've completed processing but have no response to give."),
        ([("reasoning_content", "thinking")], "I've completed processing but have no response to give."),
        ([("content", " answer ")], " answer "),
    ],
    ids=["no-chunks", "empty-content", "whitespace-content", "reasoning-only", "visible-content"],
)
async def test_final_response_is_emitted_and_persisted_when_model_returns_no_text(
    tmp_path, monkeypatch, response_chunks, expected_content,
):
    async def chunks():
        for chunk in response_chunks:
            yield chunk

    async def build_test_system_prompt(**_kwargs):
        return {"role": "system", "content": "system"}

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)
    monkeypatch.setattr(AgentLoop, "_count_context_tokens", AsyncMock(return_value=10))

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.workspace = tmp_path
    agent_loop.channel = None
    agent_loop.user_organization = None
    agent_loop.history = SimpleNamespace(load=AsyncMock(return_value=[]), save=AsyncMock())
    agent_loop.compressor = SimpleNamespace(should_compress=AsyncMock(return_value=False))
    agent_loop.chat = AsyncMock(return_value=LLMResponse(iter_mix_chunks=chunks()))
    agent_loop.records = SimpleNamespace(append=AsyncMock())
    agent_loop.tools = []
    agent_loop.max_iterations = 2
    agent_loop.max_accept_token = 100_000
    agent_loop._input_tokens = 0
    agent_loop._output_tokens = 0
    agent_loop._model_call_count = 0
    message = InboundMessage(
        user_id="user",
        session_id="session",
        sender_name="Alice",
        channel="test",
        session_metadata={},
        content=[InputText(text="hello")],
        timestamp="2026-11-05T17:53:00",
        files=[],
    )

    events = [event async for event in agent_loop.run(message)]

    assert [event for event in events if event["event"] == "assistant"] == [
        {"event": "assistant", "context": expected_content},
    ]
    agent_loop.history.save.assert_awaited_once()
    assert agent_loop.history.save.await_args.args[2][-1] == {
        "role": "assistant", "content": expected_content,
    }
    assert agent_loop.records.append.await_args.args[0].model_response == expected_content


@pytest.mark.asyncio
async def test_tool_schemas_count_toward_compression_and_context_usage(tmp_path, monkeypatch):
    async def build_test_system_prompt(**_kwargs):
        return {"role": "system", "content": "system"}

    async def search() -> str:
        return "found"

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)
    provider = LiteLLMProvider(
        api_key="test",
        api_base="https://example.test",
        default_model="test",
        parameters=LLMParameters(max_context_tokens=200),
    )
    provider.register_mcp_tool(
        search,
        tool_json={
            "type": "function",
            "function": {
                "name": "search",
                "description": "instructions " * 300,
                "parameters": {"type": "object", "properties": {}},
            },
        },
    )
    agent_loop = AgentLoop(provider=provider, workspace=tmp_path, tools=[], summary_tokens=10, max_iterations=1)
    messages = [{"role": "user", "timestamp": 1.0, "content": "hello"}]
    assert await token_counter(messages) < agent_loop.max_accept_token
    assert await agent_loop.compressor.should_compress(messages)
    assert not await agent_loop.compressor._is_within_target(messages)

    agent_loop.history = SimpleNamespace(load=AsyncMock(return_value=[]), save=AsyncMock())
    agent_loop.records = SimpleNamespace(append=AsyncMock())
    agent_loop.memory_consolidator = SimpleNamespace(consolidate=AsyncMock())
    agent_loop.compressor.compress = AsyncMock(side_effect=lambda messages: messages)
    agent_loop.chat = AsyncMock(return_value=make_response("answer"))
    message = InboundMessage(
        user_id="user",
        session_id="session",
        sender_name="Alice",
        channel="test",
        session_metadata={},
        content=[InputText(text="hello")],
        timestamp="2026-09-06T10:00:00",
        files=[],
    )

    events = [event async for event in agent_loop.run(message)]

    agent_loop.compressor.compress.assert_awaited_once()
    saved_messages = agent_loop.history.save.await_args.args[2]
    count = await agent_loop._count_context_tokens(saved_messages)
    assert 170 <= count < 1_000
    assert json.loads(events[-1]["context"])["上下文使用率"] == (f"100.00% ({count} / 170) ♻️ 即将达到最大上下文")


@pytest.mark.asyncio
@pytest.mark.parametrize("json_output", [False, True])
async def test_not_interactive_run_saves_separate_yaml_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_output):
    async def build_test_system_prompt(workspace: Path, *, channel, user_organization, not_interactive):
        assert workspace == tmp_path
        assert user_organization == "公司/研发/平台"
        assert not_interactive is True
        return {"role": "system", "content": "system"}

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)
    monkeypatch.setattr(AgentLoop, "_count_context_tokens", AsyncMock(return_value=10))

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.workspace = tmp_path
    agent_loop.channel = None
    agent_loop.user_organization = "公司/研发/平台"
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
    assert agent_loop.history.save.await_args.kwargs["channel"] == message.channel
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
    expected_not_interactive = not_interactive

    async def build_test_system_prompt(workspace: Path, *, channel, user_organization, not_interactive):
        assert workspace == tmp_path
        assert user_organization is None
        assert not_interactive is expected_not_interactive
        return {"role": "system", "content": "system"}

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.workspace = tmp_path
    agent_loop.channel = None
    agent_loop.user_organization = None
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
    assert save.await_args.kwargs["channel"] == message.channel
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


@pytest.mark.asyncio
async def test_request_yaml_retains_new_messages_across_multiple_compressions(tmp_path, monkeypatch):
    async def system_prompt(**_kwargs):
        return {"role": "system", "content": "system"}

    monkeypatch.setattr(loop_module, "build_system_prompt", system_prompt)
    monkeypatch.setattr(AgentLoop, "_count_context_tokens", AsyncMock(return_value=10))
    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.workspace = tmp_path
    agent_loop.channel = None
    agent_loop.user_organization = None
    agent_loop.history = SimpleNamespace(
        load=AsyncMock(return_value=[{"role": "user", "content": "previous question"}]),
        save=AsyncMock(),
    )
    agent_loop.compressor = SimpleNamespace(
        should_compress=AsyncMock(side_effect=[True, True]),
        compress=AsyncMock(side_effect=lambda _: [{"role": "assistant", "content": "compressed summary"}]),
    )
    agent_loop.memory_consolidator = SimpleNamespace(consolidate=AsyncMock())
    async def tool_chunks():
        yield "content", "working"
        yield "tool_calls", ToolCallChunk(index=0, id="call-1", name="search", arguments="{}")

    async def search() -> str:
        return "uncompressed tool output"

    response = LLMResponse(iter_mix_chunks=tool_chunks())
    agent_loop.chat = AsyncMock(side_effect=[response, make_response("final answer")])

    agent_loop.tool_executor = ToolExecutor(
        workspace=tmp_path, max_result_chars=10_000, mapping_tool_call_funcs={"search": search},
    )
    agent_loop.records = SimpleNamespace(append=AsyncMock())
    agent_loop.tools = []
    agent_loop.max_iterations = 4
    agent_loop.max_accept_token = 100_000
    agent_loop._input_tokens = agent_loop._output_tokens = agent_loop._model_call_count = 0
    message = InboundMessage(
        user_id="user", session_id="default", sender_name="Alice", channel="test",
        session_metadata={}, content=[InputText(text="new question")], files=[],
    )

    _events = [event async for event in agent_loop.run(message)]

    saved = agent_loop.records.append.await_args.kwargs["messages"]
    assert [item["role"] for item in saved] == ["user", "assistant", "tool", "assistant"]
    assert "new question" in str(saved[0]["content"])
    assert saved[2]["content"] == "uncompressed tool output"
    assert saved[3]["content"] == "final answer"
    assert "previous question" not in str(saved)
    assert "compressed summary" not in str(saved)
    assert agent_loop.compressor.compress.await_count == 2


@pytest.mark.asyncio
async def test_agent_run_requires_request_id(monkeypatch):
    monkeypatch.setattr(loop_module, "get_request_id", lambda: None)
    agent_loop = AgentLoop.__new__(AgentLoop)
    message = InboundMessage(
        user_id="user", session_id="default", sender_name="Alice", channel="test",
        session_metadata={}, content=[InputText(text="hello")], files=[],
    )
    with pytest.raises(RuntimeError, match="requires a request_id"):
        _events = [event async for event in agent_loop.run(message)]
