import asyncio
import json

import pytest

from xpeech.provider.schema import LLMResponse, ToolCallChunk, ToolCallRequest


@pytest.mark.asyncio
async def test_response_exposes_aggregates_only_after_stream_finishes():
    tool_call = ToolCallRequest(id="call-1", name="search", arguments={"query": "xpeech"})

    async def chunks():
        yield "reasoning_content", "think "
        yield "content", "hello "
        yield "reasoning_content", "again"
        yield "content", "world"
        yield "tool_calls", ToolCallChunk(
            index=0,
            id=tool_call.id,
            name=tool_call.name,
            arguments=json.dumps(tool_call.arguments, ensure_ascii=False),
        )

    response = LLMResponse(iter_mix_chunks=chunks())

    with pytest.raises(ValueError, match="Content is not available"):
        _ = response.content
    with pytest.raises(ValueError, match="Reasoning is not available"):
        _ = response.reasoning
    with pytest.raises(ValueError, match="Tool calls are not available"):
        _ = response.tool_calls

    streamed = [chunk async for chunk in response.iter_mix_chunks]

    assert streamed == [
        ("reasoning_content", "think "),
        ("reasoning_content_end", None),
        ("content", "hello "),
        ("content_end", None),
        ("reasoning_content", "again"),
        ("reasoning_content_end", None),
        ("content", "world"),
        ("content_end", None),
        (
            "tool_calls",
            ToolCallChunk(
                index=0,
                id="call-1",
                name="search",
                arguments='{"query": "xpeech"}',
            ),
        ),
        ("tool_calls_end", None),
    ]
    assert response.content == "hello world"
    assert response.reasoning == "think again"
    assert response.tool_calls == [tool_call]
    assert response.has_tool_calls


@pytest.mark.asyncio
async def test_flush_drains_stream():
    chunk_seen = asyncio.Event()

    async def chunks():
        yield "content", "one"
        chunk_seen.set()
        await asyncio.sleep(0)
        yield "content", "two"

    response = LLMResponse(iter_mix_chunks=chunks())
    flush_task = asyncio.create_task(response.flush())

    await chunk_seen.wait()
    assert await flush_task is response
    assert response.content == "onetwo"


@pytest.mark.asyncio
async def test_stopping_consumption_early_does_not_mark_stream_done():
    async def chunks():
        yield "content", "one"
        yield "content", "two"

    response = LLMResponse(iter_mix_chunks=chunks())

    assert await anext(response.iter_mix_chunks) == ("content", "one")
    with pytest.raises(ValueError, match="Content is not available"):
        _ = response.content

    await response.flush()
    assert response.content == "onetwo"


@pytest.mark.asyncio
async def test_tool_calls_merges_parallel_deltas_by_index():
    async def chunks():
        for chunk in [
            ToolCallChunk(index=0, id="call-1", name="get_weather", arguments=""),
            ToolCallChunk(index=0, arguments='{"city": "'),
            ToolCallChunk(index=0, arguments="北京"),
            ToolCallChunk(index=0, arguments='"}'),
            ToolCallChunk(index=1, id="call-2", name="get_local_time", arguments=""),
            ToolCallChunk(index=1, arguments='{"city": "'),
            ToolCallChunk(index=1, arguments="上海"),
            ToolCallChunk(index=1, arguments='"}'),
        ]:
            yield "tool_calls", chunk

    response = LLMResponse(iter_mix_chunks=chunks())
    await response.flush()

    assert response.tool_calls == [
        ToolCallRequest(id="call-1", name="get_weather", arguments={"city": "北京"}),
        ToolCallRequest(id="call-2", name="get_local_time", arguments={"city": "上海"}),
    ]
