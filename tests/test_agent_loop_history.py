import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session, make_transient_to_detached
from sqlalchemy.orm.exc import DetachedInstanceError

import xpeech.agent.loop as loop_module
from xpeech.agent.loop import AgentLoop
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.provider.openai_provider import OpenAIProvider
from xpeech.provider.schema import LLMParameters, LLMResponse, ToolCallChunk
from xpeech.utils.helper import token_counter

pytestmark = pytest.mark.usefixtures("fake_vllm_tokenizer")


@pytest.fixture(autouse=True)
def request_context(monkeypatch):
    monkeypatch.setattr(loop_module, "build_context_messages", AsyncMock(return_value=[]))
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
        ([("reasoning", "thinking")], "I've completed processing but have no response to give."),
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
    agent_loop.provider = SimpleNamespace(
        default_context_token=200_000, default_model="test",
        chat_count_tool_tokens=AsyncMock(return_value={"system_tools": 0, "mcp_tools": 0}),
    )
    agent_loop.workspace = tmp_path
    agent_loop.channel = None
    agent_loop.user_organization = None
    agent_loop.history = SimpleNamespace(load=AsyncMock(return_value=[]), save=AsyncMock())
    agent_loop.compressor = SimpleNamespace(should_compress=AsyncMock(return_value=False))
    agent_loop.chat = AsyncMock(return_value=LLMResponse(iter_mix_chunks=chunks()))
    agent_loop.records = SimpleNamespace(append=AsyncMock())
    agent_loop.tools = []
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

    thinking_events = [event for event in events if event["event"] in {"thinking", "thinking_end"}]
    assert thinking_events == ([
        {"event": "thinking", "context": "thinking"},
        {"event": "thinking_end", "context": ""},
    ] if ("reasoning", "thinking") in response_chunks else [])
    assert [event for event in events if event["event"] == "assistant"] == [
        {"event": "assistant", "context": expected_content},
    ]
    agent_loop.history.save.assert_awaited_once()
    assert agent_loop.history.save.await_args.args[2][-1] == {
        "role": "assistant", "content": expected_content,
    }
    assert agent_loop.records.append.await_args.args[0].model_response == expected_content


@pytest.mark.asyncio
async def test_token_usage_is_emitted_after_saved_record_expires(tmp_path, make_openai_sdk_client):
    response = make_response("answer")
    response.usage.update({"prompt_tokens": 1234, "completion_tokens": 5678})
    provider = OpenAIProvider(
        client=make_openai_sdk_client(), default_model="test", parameters=LLMParameters(),
    )
    provider.chat = AsyncMock(return_value=response)
    agent_loop = AgentLoop(provider=provider, workspace=tmp_path, tools=[])
    agent_loop.history = SimpleNamespace(load=AsyncMock(return_value=[]), save=AsyncMock())

    async def save_record(record, *, messages):
        # 重现提交后字段过期、会话关闭后的真实 ORM 状态，无需连接数据库。
        record.id = 1
        make_transient_to_detached(record)
        with Session() as session:
            session.add(record)
            session.expire(record)

    agent_loop.records = SimpleNamespace(append=AsyncMock(side_effect=save_record))
    message = InboundMessage(
        user_id="user", session_id="session", sender_name="Alice", channel="test",
        session_metadata={}, content=[InputText(text="hello")], files=[],
    )

    events = [event async for event in agent_loop.run(message)]

    with pytest.raises(DetachedInstanceError):
        _ = agent_loop.records.append.await_args.args[0].input_tokens
    assert {"event": "assistant", "context": "answer"} in events
    assert events[-1]["event"] == "token_usage"
    metrics = json.loads(events[-1]["context"])
    assert (metrics["输入 Token"], metrics["输出 Token"], metrics["总 Token"]) == ("1k", "5k", "6k")
    assert metrics["大模型请求次数"] == "1"
    assert "上下文明细" in metrics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context_tokens, output_tokens, expected_threshold",
    [(200000, 32768, 147000), (262144, 32768, 202929), (200000, 20001, 147000)],
)
async def test_compression_uses_90_percent_of_context_window(
    tmp_path, monkeypatch, context_tokens, output_tokens, expected_threshold, make_openai_sdk_client,
):
    count_tokens = AsyncMock(return_value=expected_threshold - 1)
    monkeypatch.setattr(AgentLoop, "_count_context_tokens", count_tokens)
    provider = OpenAIProvider(
        client=make_openai_sdk_client(),
        default_model="test",
        parameters=LLMParameters(max_context_tokens=context_tokens, max_tokens=output_tokens),
    )
    agent_loop = AgentLoop(provider=provider, workspace=tmp_path, tools=[])

    assert agent_loop.max_accept_token == expected_threshold
    assert agent_loop.compressor._target_tokens == expected_threshold // 2
    assert not await agent_loop.compressor.should_compress([])
    count_tokens.return_value = expected_threshold
    assert await agent_loop.compressor.should_compress([])


@pytest.mark.asyncio
async def test_tool_schemas_count_toward_compression_and_context_usage(tmp_path, monkeypatch, make_openai_sdk_client):
    async def build_test_system_prompt(**_kwargs):
        return {"role": "system", "content": "system"}

    async def search() -> str:
        return "found"

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)
    provider = OpenAIProvider(
        client=make_openai_sdk_client(),
        default_model="test",
        parameters=LLMParameters(max_context_tokens=200000),
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
    agent_loop = AgentLoop(provider=provider, workspace=tmp_path, tools=[])
    # 使用较小预算验证工具定义也会触发压缩。
    agent_loop.max_accept_token = 170
    agent_loop.compressor = loop_module.ConversationCompressor(
        chat=agent_loop.chat,
        max_accept_tokens=170,
        target_tokens=85,
        token_counter=agent_loop._count_context_tokens,
    )
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
    # 系统提示词、对话和工具定义独立计数后直接累计。
    context_count = (
        await token_counter([saved_messages[0]])
        + await token_counter(loop_module.strip_internal_message_metadata(saved_messages[1:]))
        + await token_counter(tools=[provider.default_tools["search"].tool_json])
    )
    usage = json.loads(events[-1]["context"])
    assert usage["上下文使用率"] == (f"{min(1, context_count / 170):.2%} ({context_count // 1000}k / 0k) ♻️ 即将达到最大上下文")
    details = usage["上下文明细"]
    parts = {name: part for name, part in details.items() if name != "剩余空间"}
    assert details["MCP 工具"]["词元数"] > 0
    assert sum(part["词元数"] for part in parts.values()) == context_count
    assert all(part["占比"] == min(1, part["词元数"] / 170) for part in parts.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("json_output", [False, True])
async def test_not_interactive_run_saves_yaml_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_output):
    async def build_test_context_messages(
        workspace: Path, *, user_id, channel, user_organization, not_interactive, token_counts
    ):
        assert workspace == tmp_path
        assert user_id == "session-1"
        assert user_organization == "公司/研发/平台"
        assert not_interactive is True
        return []

    monkeypatch.setattr(loop_module, "build_context_messages", build_test_context_messages)
    monkeypatch.setattr(loop_module, "build_system_prompt", AsyncMock(return_value={"role": "system", "content": "system"}))
    monkeypatch.setattr(AgentLoop, "_count_context_tokens", AsyncMock(return_value=10))

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.provider = SimpleNamespace(
        default_context_token=200_000, default_model="test",
        chat_count_tool_tokens=AsyncMock(return_value={"system_tools": 0, "mcp_tools": 0}),
    )
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
    assert agent_loop.history.save.await_args.kwargs == {"channel": message.channel}
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

    async def build_test_context_messages(
        workspace: Path, *, user_id, channel, user_organization, not_interactive, token_counts
    ):
        assert workspace == tmp_path
        assert user_id == "session-1"
        assert user_organization is None
        assert not_interactive is expected_not_interactive
        return []

    monkeypatch.setattr(loop_module, "build_context_messages", build_test_context_messages)
    monkeypatch.setattr(loop_module, "build_system_prompt", AsyncMock(return_value={"role": "system", "content": "system"}))

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.provider = SimpleNamespace(
        chat_count_tool_tokens=AsyncMock(return_value={"system_tools": 0, "mcp_tools": 0}),
    )
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
    assert save.await_args.kwargs == {"channel": message.channel}
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
    agent_loop.provider = SimpleNamespace(
        default_context_token=200_000, default_model="test",
        chat_count_tool_tokens=AsyncMock(return_value={"system_tools": 0, "mcp_tools": 0}),
    )
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
    agent_loop.max_accept_token = 100_000
    agent_loop._input_tokens = agent_loop._output_tokens = agent_loop._model_call_count = 0
    message = InboundMessage(
        user_id="user", session_id="00000000-0000-7000-8000-000000000000", sender_name="Alice", channel="test",
        session_metadata={}, content=[InputText(text="new question")], files=[],
    )

    _events = [event async for event in agent_loop.run(message)]

    saved = agent_loop.records.append.await_args.kwargs["messages"]
    assert [item["role"] for item in saved] == ["user", "assistant", "tool", "assistant"]
    assert "new question" in str(saved[0]["content"])
    assert saved[2]["content"].startswith("【工具调用】")
    assert saved[2]["content"].split("\n\n", 1)[1] == "uncompressed tool output"
    assert saved[3]["content"] == "final answer"
    assert "previous question" not in str(saved)
    assert "compressed summary" not in str(saved)
    assert agent_loop.compressor.compress.await_count == 2


@pytest.mark.asyncio
async def test_agent_run_requires_request_id(monkeypatch):
    monkeypatch.setattr(loop_module, "get_request_id", lambda: None)
    agent_loop = AgentLoop.__new__(AgentLoop)
    message = InboundMessage(
        user_id="user", session_id="00000000-0000-7000-8000-000000000000", sender_name="Alice", channel="test",
        session_metadata={}, content=[InputText(text="hello")], files=[],
    )
    with pytest.raises(RuntimeError, match="requires a request_id"):
        _events = [event async for event in agent_loop.run(message)]
