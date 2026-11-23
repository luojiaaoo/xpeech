import asyncio
import inspect
import threading
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from xpeech.provider.openai_provider import OpenAIProvider
from xpeech.provider.schema import LLMParameters, RegisteredTool, ToolCallChunk, ToolCallRequest
from xpeech.utils.helper import token_counter

pytestmark = pytest.mark.usefixtures("fake_vllm_tokenizer")


@pytest.mark.asyncio
@pytest.mark.parametrize("override_mcp", [False, True])
async def test_tool_usage_counts_only_effective_schemas(override_mcp, make_openai_sdk_client):
    async def local_tool() -> str:
        """Run a local operation."""
        return "local"

    async def remote_tool() -> str:
        """Override the remote tool locally."""
        return "remote"

    provider = OpenAIProvider(
        client=make_openai_sdk_client(), default_model="test", parameters=LLMParameters(),
    )
    provider.register_function_tool(local_tool)
    provider.register_mcp_tool(remote_tool, tool_json={
        "type": "function", "function": {
            "name": "remote_tool", "description": "MCP instructions " * 100,
            "parameters": {"type": "object", "properties": {}},
        },
    })
    tools = [remote_tool] if override_mcp else []
    counts = await provider.chat_count_tool_tokens(tools)
    enabled = await provider._resolve_tools(tools)
    local_schemas = [tool.tool_json for tool in enabled.values() if tool.source == "system_tools"]
    assert counts["system_tools"] == await token_counter(tools=local_schemas)
    if not override_mcp:
        assert counts["mcp_tools"] == await token_counter(tools=[provider.default_tools["remote_tool"].tool_json])
    assert counts["system_tools"] > 0
    assert (counts["mcp_tools"] == 0) is override_mcp

    empty = OpenAIProvider(
        client=make_openai_sdk_client(), default_model="test", parameters=LLMParameters(),
    )
    assert await empty.chat_count_tool_tokens() == {"system_tools": 0, "mcp_tools": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["break", "cancel_flush"])
async def test_closing_parsed_model_stream_returns_permit_without_task_exit(
    make_openai_sdk_client, exit_kind, model_permits,
):
    reading = asyncio.Event()
    closed = asyncio.Event()

    async def upstream():
        try:
            yield stream_chunk(delta=SimpleNamespace(content="partial", reasoning=None, tool_calls=None))
            reading.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    sdk_stream = upstream()  # 保留引用，验证关闭链不依赖 GC。

    provider = OpenAIProvider(
        client=make_openai_sdk_client(AsyncMock(return_value=sdk_stream)),
        default_model="test",
        parameters=LLMParameters(),
    )
    response = await provider.chat(messages=[])
    completed = []
    response.add_completion_callback(completed.append)
    if exit_kind == "break":
        async with aclosing(response.iter_mix_chunks) as chunks:
            async for chunk in chunks:
                assert chunk == ("content", "partial")
                assert await model_permits.count() == 1
                break
    else:
        task = asyncio.create_task(response.flush())
        await asyncio.wait_for(reading.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()
    await model_permits.wait_for(0)
    assert completed == []
    with pytest.raises(ValueError, match="Content is not available"):
        _ = response.content


def stream_chunk(*, delta=None, finish_reason=None, usage=None):
    choices = []
    if delta is not None or finish_reason is not None:
        choices.append(SimpleNamespace(delta=delta, finish_reason=finish_reason))
    return SimpleNamespace(choices=choices, usage=usage)


def test_chat_exposes_llm_parameter_overrides_as_one_object():
    parameters = inspect.signature(OpenAIProvider.chat).parameters

    assert "parameters" in parameters
    assert "model" not in parameters
    assert "max_tokens" not in parameters
    assert "top_p" not in parameters
    assert "reasoning_effort" not in parameters
    assert "remove_interactive_tool" not in parameters
    assert "is_interactive" not in inspect.signature(OpenAIProvider.register_function_tool).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "expected_names"),
    [
        ({}, ["search", "default_tool", "extra_tool"]),
        ({"remove_default_tools": True}, ["search", "extra_tool"]),
        ({"remove_all_tools": True}, []),
        ({"remove_default_tools": True, "remove_all_tools": True}, []),
    ],
)
async def test_removal_options_apply_only_to_chat(monkeypatch, options, expected_names, make_openai_sdk_client):
    captured_kwargs = {}

    class StreamClient:
        def create(self, **kwargs):
            captured_kwargs.clear()
            captured_kwargs.update(kwargs)

            async def upstream():
                yield stream_chunk(finish_reason="stop")

            return upstream()

    async def search() -> str:
        """Search using the overridden schema."""
        return "found"

    async def default_tool() -> str:
        """A default tool."""
        return "default"

    async def extra_tool() -> str:
        """An extra tool."""
        return "extra"

    monkeypatch.setattr("xpeech.provider.openai_provider.get_custom_tool_func", lambda name: search)
    provider = OpenAIProvider(
        client=make_openai_sdk_client(), default_model="test", parameters=LLMParameters()
    )
    provider.register_mcp_tool(
        search,
        tool_json={
            "type": "function",
            "function": {
                "name": "search",
                "description": "Original MCP schema",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    )
    provider.register_function_tool(default_tool)
    provider._stream_client = StreamClient()
    messages = [{"role": "user", "content": "hello"}]
    request = {"messages": messages, "tools": ["search", extra_tool], **options}

    # 压缩预算保持整包计数，不接受 chat 的 remove_* 开关。
    count = await provider.chat_count_tokens(messages=messages, tools=["search", extra_tool])
    await provider.chat(messages=messages, tools=["search", extra_tool])
    full_tools = captured_kwargs["tools"]
    await provider.chat(**request)

    actual_tools = captured_kwargs.get("tools")
    assert [tool["function"]["name"] for tool in actual_tools or []] == expected_names
    assert count == await token_counter(messages, tools=full_tools)
    if expected_names:
        assert count > await token_counter(messages)
        assert actual_tools[0]["function"]["description"] == search.__doc__
    assert provider.default_tools["search"].tool_json["function"]["description"] == "Original MCP schema"


@pytest.mark.asyncio
async def test_registered_tools_keep_schema_and_callable(make_openai_sdk_client):
    provider = OpenAIProvider(
        client=make_openai_sdk_client(),
        default_model="test",
        parameters=LLMParameters(),
    )
    event_loop_thread = threading.get_ident()

    def interactive_tool() -> str:
        """Return the worker thread identifier."""
        return str(threading.get_ident())

    def non_interactive_tool() -> str:
        """Return the current thread identifier."""
        return str(threading.get_ident())

    async def async_tool() -> str:
        """Return the event-loop thread identifier."""
        return str(threading.get_ident())

    assert provider.register_function_tool(interactive_tool) is None
    assert provider.register_function_tool(non_interactive_tool) is None
    assert provider.register_function_tool(async_tool) is None

    assert set(provider.default_tools) == {"interactive_tool", "non_interactive_tool", "async_tool"}
    assert all(isinstance(tool, RegisteredTool) for tool in provider.default_tools.values())
    assert provider.default_tools["interactive_tool"].tool_json["function"]["name"] == "interactive_tool"
    assert await provider.default_tools["interactive_tool"].func() != str(event_loop_thread)
    assert await provider.default_tools["non_interactive_tool"].func() != str(event_loop_thread)
    assert await provider.default_tools["async_tool"].func() == str(event_loop_thread)


@pytest.mark.asyncio
async def test_mcp_tools_keep_schema_and_callable(make_openai_sdk_client):
    async def mcp_tool() -> str:
        return "ok"

    tool_json = {"type": "function", "function": {"name": "mcp_tool"}}
    provider = OpenAIProvider(
        client=make_openai_sdk_client(),
        default_model="test",
        parameters=LLMParameters(),
    )

    result = provider.register_mcp_tool(mcp_tool, tool_json=tool_json)

    assert result is None
    assert provider.default_tools["mcp_tool"].func.__wrapped__ is mcp_tool
    assert provider.default_tools["mcp_tool"].tool_json is tool_json
    assert await provider.default_tools["mcp_tool"].func() == "ok"


@pytest.mark.asyncio
async def test_chat_uses_all_registered_tools(make_openai_sdk_client):
    captured_kwargs = None

    async def upstream():
        yield stream_chunk(
            delta=SimpleNamespace(reasoning=None, content="ok", tool_calls=None),
            finish_reason="stop",
        )

    class StreamClient:
        def create(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return upstream()

    async def interactive_tool() -> str:
        """An interactive tool."""
        return "interactive"

    async def regular_tool() -> str:
        """A regular tool."""
        return "regular"

    provider = OpenAIProvider(
        client=make_openai_sdk_client(),
        default_model="test",
        parameters=LLMParameters(),
    )
    provider._stream_client = StreamClient()
    provider.register_function_tool(interactive_tool)
    provider.register_function_tool(regular_tool)

    await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
    )

    assert captured_kwargs is not None
    assert [tool["function"]["name"] for tool in captured_kwargs["tools"]] == ["interactive_tool", "regular_tool"]
    assert set(provider.mapping_tool_call_funcs) == {"interactive_tool", "regular_tool"}
    assert set(provider.default_tools) == {"interactive_tool", "regular_tool"}

    await provider.chat(messages=[{"role": "user", "content": "hello"}])

    assert [tool["function"]["name"] for tool in captured_kwargs["tools"]] == [
        "interactive_tool", "regular_tool"
    ]
    assert set(provider.mapping_tool_call_funcs) == {"interactive_tool", "regular_tool"}


@pytest.mark.asyncio
async def test_parse_response_streams_text_and_assembles_tool_calls(make_openai_sdk_client):
    async def upstream():
        yield stream_chunk(delta=ChoiceDelta.model_validate({"reasoning": "why "}))
        yield stream_chunk(delta=SimpleNamespace(reasoning=None, content="hello ", tool_calls=None))
        yield stream_chunk(
            delta=SimpleNamespace(
                reasoning=None,
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call-1",
                        function=SimpleNamespace(name="search", arguments='{"query":'),
                    )
                ],
            )
        )
        yield stream_chunk(
            delta=SimpleNamespace(
                reasoning=None,
                content="world",
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(name=None, arguments='"xpeech"}'),
                    )
                ],
            ),
            finish_reason="tool_calls",
        )
        yield stream_chunk(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )

    provider = OpenAIProvider(
        client=make_openai_sdk_client(),
        default_model="test",
        parameters=LLMParameters(),
    )
    response = provider._parse_response(upstream())
    assert response.finish_reason is None

    chunks = [chunk async for chunk in response.iter_mix_chunks]

    assert chunks == [
        ("reasoning", "why "),
        ("reasoning_end", None),
        ("content", "hello "),
        ("content_end", None),
        ("tool_calls", ToolCallChunk(index=0, id="call-1", name="search", arguments='{"query":')),
        ("tool_calls_end", None),
        ("content", "world"),
        ("content_end", None),
        ("tool_calls", ToolCallChunk(index=0, arguments='"xpeech"}')),
        ("tool_calls_end", None),
    ]
    assert response.reasoning == "why "
    assert response.content == "hello world"
    assert response.tool_calls == [
        ToolCallRequest(id="call-1", name="search", arguments={"query": "xpeech"})
    ]
    assert response.finish_reason == "tool_calls"
    assert response.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


@pytest.mark.asyncio
async def test_chat_delegates_stream_defaults_to_stream_client(make_openai_sdk_client):
    captured_kwargs = None

    async def upstream():
        yield stream_chunk(
            delta=SimpleNamespace(reasoning=None, content="ok", tool_calls=None),
            finish_reason="stop",
        )

    class StreamClient:
        def create(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return upstream()

    parameters = LLMParameters(
        max_tokens=4096,
        max_context_tokens=262144,
        temperature=0.8,
        top_p=0.9,
        top_k=10,
        min_p=0.1,
        presence_penalty=0.2,
        repetition_penalty=1.1,
        reasoning_effort="high",
    )
    provider = OpenAIProvider(
        client=make_openai_sdk_client(),
        default_model="test",
        parameters=parameters,
    )
    provider._stream_client = StreamClient()

    response = await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        parameters=LLMParameters(max_tokens=1024),
    )
    await response.flush()

    assert captured_kwargs is not None
    assert "stream" not in captured_kwargs
    assert "stream_options" not in captured_kwargs
    assert captured_kwargs["max_tokens"] == 1024
    assert captured_kwargs["temperature"] == 0.8
    assert captured_kwargs["top_p"] == 0.9
    assert captured_kwargs["presence_penalty"] == 0.2
    assert captured_kwargs["reasoning_effort"] == "high"
    assert captured_kwargs["extra_body"] == {
        "top_k": 10,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
    }
    assert provider.default_context_token == 262144
    assert response.content == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("json_output,support_json_output", [(False, False), (False, True), (True, False), (True, True)])
async def test_chat_omits_unset_optional_parameters(json_output, support_json_output, make_openai_sdk_client):
    captured_kwargs = None

    async def upstream():
        yield stream_chunk(
            delta=SimpleNamespace(reasoning=None, content="ok", tool_calls=None),
            finish_reason="stop",
        )

    class StreamClient:
        def create(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return upstream()

    provider = OpenAIProvider(
        client=make_openai_sdk_client(),
        default_model="test",
        parameters=LLMParameters(),
        support_json_output=support_json_output,
    )
    provider._stream_client = StreamClient()

    if json_output and not support_json_output:
        with pytest.raises(HTTPException) as exc_info:
            await provider.chat(messages=[{"role": "user", "content": "hello"}], json_output=True)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "The configured model does not support JSON output"
        assert captured_kwargs is None
        return

    response = await provider.chat(messages=[{"role": "user", "content": "hello"}], json_output=json_output)
    await response.flush()

    assert captured_kwargs is not None
    assert captured_kwargs["max_tokens"] == 32768
    assert captured_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    if json_output and support_json_output:
        assert captured_kwargs["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in captured_kwargs
    for parameter in (
        "temperature",
        "top_p",
        "presence_penalty",
        "extra_body",
        "extra_headers",
        "reasoning_effort",
    ):
        assert parameter not in captured_kwargs
