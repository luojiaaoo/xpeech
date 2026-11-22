import asyncio
from contextlib import aclosing

import pytest

from xpeech.config.settings import settings
from xpeech.provider import openai_stream
from xpeech.provider.openai_stream import OpenAIStreamClient


class FakeStreamError(Exception):
    pass


def test_format_error_exposes_only_status_and_message():
    class InternalProviderFunctionName(Exception):
        status_code = 400

        def __str__(self):
            return "The request payload is invalid"

    detail = openai_stream._format_error(InternalProviderFunctionName())

    assert detail == "error_code=400, error_message=The request payload is invalid"
    assert "InternalProviderFunctionName" not in detail


@pytest.mark.asyncio
async def test_create_always_streams_with_usage(make_openai_sdk_client):
    captured_kwargs = None

    async def fake_create(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs

        async def stream():
            yield "ok"

        return stream()

    client = OpenAIStreamClient(make_openai_sdk_client(fake_create))
    response = client.create(model="test", stream=False)

    assert [chunk async for chunk in response] == ["ok"]
    assert captured_kwargs["model"] == "test"
    assert captured_kwargs["stream"] is True
    assert captured_kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_does_not_retry_errors_raised_during_iteration(make_openai_sdk_client):
    calls = 0

    async def fake_create(**_kwargs):
        nonlocal calls
        calls += 1

        async def stream():
            raise FakeStreamError("stream failed")
            yield

        return stream()

    client = OpenAIStreamClient(make_openai_sdk_client(fake_create))
    with pytest.raises(FakeStreamError, match="stream failed"):
        await _collect(client.create())

    assert calls == 1


@pytest.mark.asyncio
async def test_stream_propagates_non_rate_limit_errors(make_openai_sdk_client):
    async def fake_create(**_kwargs):
        async def stream():
            raise ValueError("provider rejected the request")
            yield

        return stream()

    client = OpenAIStreamClient(make_openai_sdk_client(fake_create))
    with pytest.raises(ValueError, match="provider rejected the request") as exc_info:
        await _collect(client.create())

    assert str(exc_info.value) == "provider rejected the request"


@pytest.mark.asyncio
async def test_stream_holds_redis_permit_until_consumption_finishes(monkeypatch, make_openai_sdk_client):
    monkeypatch.setattr(settings.llm, "parallel", 1)
    first_stream_started = asyncio.Event()
    release_first_stream = asyncio.Event()
    calls = 0
    active_streams = 0
    max_active_streams = 0

    async def fake_create(**kwargs):
        nonlocal calls
        calls += 1
        call_number = calls

        async def stream():
            nonlocal active_streams, max_active_streams
            active_streams += 1
            max_active_streams = max(max_active_streams, active_streams)
            try:
                if call_number == 1:
                    first_stream_started.set()
                    await release_first_stream.wait()
                yield call_number
            finally:
                active_streams -= 1

        return stream()

    client = OpenAIStreamClient(make_openai_sdk_client(fake_create))
    first = client.create()
    second = client.create()

    first_task = asyncio.create_task(_collect(first))
    await first_stream_started.wait()
    second_task = asyncio.create_task(_collect(second))
    await asyncio.sleep(0)

    assert calls == 1
    assert max_active_streams == 1

    release_first_stream.set()
    assert await first_task == [1]
    assert await second_task == [2]
    assert calls == 2
    assert max_active_streams == 1


async def _collect(stream):
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_cancel_while_opening_completion_returns_permit(make_openai_sdk_client, model_permits):
    started = asyncio.Event()

    async def fake_create(**kwargs):
        started.set()
        await asyncio.Event().wait()

    stream = OpenAIStreamClient(make_openai_sdk_client(fake_create)).create()
    task = asyncio.create_task(_collect(stream))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await model_permits.wait_for(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["normal", "error", "timeout"])
async def test_close_returns_permit_before_sdk_cleanup(monkeypatch, cleanup, make_openai_sdk_client, model_permits):
    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    class SDKStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return "chunk"

        async def aclose(self):
            close_started.set()
            if cleanup == "error":
                raise RuntimeError("HTTP cleanup failed")
            await finish_close.wait()

    async def fake_create(**kwargs):
        return SDKStream()

    monkeypatch.setattr(openai_stream, "STREAM_CLOSE_TIMEOUT", 0.05)
    stream = OpenAIStreamClient(make_openai_sdk_client(fake_create)).create()
    assert await anext(stream) == "chunk"
    assert await model_permits.count() == 1

    close_task = asyncio.create_task(stream.aclose())
    await asyncio.wait_for(close_started.wait(), timeout=1)
    await model_permits.wait_for(0)
    if cleanup == "normal":
        finish_close.set()
    await asyncio.wait_for(close_task, timeout=1)


@pytest.mark.asyncio
async def test_aclosing_and_later_stream_close_release_exactly_once(make_openai_sdk_client, model_permits):
    closed = asyncio.Event()
    close_count = 0

    class SDKStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return "chunk"

        async def aclose(self):
            nonlocal close_count
            close_count += 1
            closed.set()

    async def fake_create(**kwargs):
        return SDKStream()

    stream = OpenAIStreamClient(make_openai_sdk_client(fake_create)).create()

    async def consume_one_chunk():
        async with aclosing(stream):
            assert await anext(stream) == "chunk"

    await asyncio.create_task(consume_one_chunk())
    await asyncio.wait_for(closed.wait(), timeout=1)
    await stream.aclose()

    assert close_count == 1
    await model_permits.wait_for(0)


@pytest.mark.asyncio
async def test_sdk_shield_and_suppressed_cancellation_cannot_extend_close_deadline(monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    finish = asyncio.Event()
    closed = asyncio.Event()

    class SlowStream:
        async def aclose(self):
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.set()
                # 模拟 SDK 吞掉取消后继续等待清理。
                await finish.wait()
            finally:
                closed.set()

    stream = SlowStream()
    monkeypatch.setattr(openai_stream, "STREAM_CLOSE_TIMEOUT", 0.02)
    task = asyncio.create_task(openai_stream._close_stream(stream))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        done, _ = await asyncio.wait({task}, timeout=0.3)
        assert task in done  # SDK 的 shield 和取消处理不能延长调用方等待时间。
        await task
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        assert not closed.is_set()
        assert openai_stream._STREAM_CLOSE_TASKS  # 未结束的清理任务仍有引用。
    finally:
        finish.set()
        await asyncio.wait_for(task, timeout=1)
        await asyncio.wait_for(closed.wait(), timeout=1)
        if openai_stream._STREAM_CLOSE_TASKS:
            await asyncio.gather(*tuple(openai_stream._STREAM_CLOSE_TASKS), return_exceptions=True)
        await asyncio.sleep(0)  # 等待 Task 的 done callback 移除保留引用。
    assert not openai_stream._STREAM_CLOSE_TASKS


@pytest.mark.asyncio
async def test_close_stream_uses_native_openai_close():
    closed = asyncio.Event()

    class Stream:
        async def close(self):
            closed.set()

    await openai_stream._close_stream(Stream())

    assert closed.is_set()


@pytest.mark.asyncio
async def test_cancelling_close_wait_also_cancels_sdk_cleanup():
    started = asyncio.Event()
    closed = asyncio.Event()

    class Stream:
        async def aclose(self):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

    task = asyncio.create_task(openai_stream._close_stream(Stream()))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(closed.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not openai_stream._STREAM_CLOSE_TASKS
