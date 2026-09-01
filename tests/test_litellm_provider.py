import inspect
from types import SimpleNamespace

import pytest

from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import LLMParameters, ToolCallChunk, ToolCallRequest


def stream_chunk(*, delta=None, finish_reason=None, usage=None):
    choices = []
    if delta is not None or finish_reason is not None:
        choices.append(SimpleNamespace(delta=delta, finish_reason=finish_reason))
    return SimpleNamespace(choices=choices, usage=usage)


def test_chat_exposes_llm_parameter_overrides_as_one_object():
    parameters = inspect.signature(LiteLLMProvider.chat).parameters

    assert "parameters" in parameters
    assert "model" not in parameters
    assert "max_tokens" not in parameters
    assert "top_p" not in parameters
    assert "reasoning_effort" not in parameters


@pytest.mark.asyncio
async def test_parse_response_streams_text_and_assembles_tool_calls():
    async def upstream():
        yield stream_chunk(delta=SimpleNamespace(reasoning_content="why ", content=None, tool_calls=None))
        yield stream_chunk(delta=SimpleNamespace(reasoning_content=None, content="hello ", tool_calls=None))
        yield stream_chunk(
            delta=SimpleNamespace(
                reasoning_content=None,
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
                reasoning_content=None,
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

    provider = LiteLLMProvider(
        api_key="test",
        api_base="https://example.test",
        default_model="test",
        parameters=LLMParameters(),
    )
    response = provider._parse_response(upstream(), {})
    assert response.finish_reason is None

    chunks = [chunk async for chunk in response.iter_mix_chunks]

    assert chunks == [
        ("reasoning_content", "why "),
        ("reasoning_content_end", None),
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
async def test_chat_requests_litellm_streaming():
    captured_kwargs = None

    async def upstream():
        yield stream_chunk(
            delta=SimpleNamespace(reasoning_content=None, content="ok", tool_calls=None),
            finish_reason="stop",
        )

    class RetryClient:
        async def acompletion(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return upstream()

    parameters = LLMParameters(
        max_tokens=4096,
        max_context_tokens=65536,
        temperature=0.8,
        top_p=0.9,
        top_k=10,
        min_p=0.1,
        presence_penalty=0.2,
        repetition_penalty=1.1,
        reasoning_effort="high",
    )
    provider = LiteLLMProvider(
        api_key="test",
        api_base="https://example.test",
        default_model="test",
        parameters=parameters,
    )
    provider._retry_client = RetryClient()

    response = await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        parameters=LLMParameters(max_tokens=1024),
    )
    await response.flush()

    assert captured_kwargs is not None
    assert captured_kwargs["stream"] is True
    assert captured_kwargs["stream_options"] == {"include_usage": True}
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
    assert provider.default_context_token == 65536
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_chat_omits_unset_optional_parameters():
    captured_kwargs = None

    async def upstream():
        yield stream_chunk(
            delta=SimpleNamespace(reasoning_content=None, content="ok", tool_calls=None),
            finish_reason="stop",
        )

    class RetryClient:
        async def acompletion(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return upstream()

    provider = LiteLLMProvider(
        api_key="test",
        api_base="https://example.test",
        default_model="test",
        parameters=LLMParameters(),
    )
    provider._retry_client = RetryClient()

    response = await provider.chat(messages=[{"role": "user", "content": "hello"}])
    await response.flush()

    assert captured_kwargs is not None
    assert captured_kwargs["max_tokens"] == 32768
    for parameter in (
        "temperature",
        "top_p",
        "presence_penalty",
        "extra_body",
        "response_format",
        "extra_headers",
        "reasoning_effort",
    ):
        assert parameter not in captured_kwargs
