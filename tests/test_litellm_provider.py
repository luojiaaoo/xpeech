from types import SimpleNamespace

import pytest

from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import ToolCallChunk, ToolCallRequest


def stream_chunk(*, delta=None, finish_reason=None, usage=None):
    choices = []
    if delta is not None or finish_reason is not None:
        choices.append(SimpleNamespace(delta=delta, finish_reason=finish_reason))
    return SimpleNamespace(choices=choices, usage=usage)


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

    provider = LiteLLMProvider(api_key="test", api_base="https://example.test", default_model="test")
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

    provider = LiteLLMProvider(api_key="test", api_base="https://example.test", default_model="test")
    provider._retry_client = RetryClient()

    response = await provider.chat(messages=[{"role": "user", "content": "hello"}])
    await response.flush()

    assert captured_kwargs is not None
    assert captured_kwargs["stream"] is True
    assert captured_kwargs["stream_options"] == {"include_usage": True}
    assert response.content == "ok"
