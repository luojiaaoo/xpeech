import asyncio
from types import SimpleNamespace

import pytest

from xpeech.provider import helper
from xpeech.provider.helper import LiteLLMRetryClient


class FakeRateLimitError(Exception):
    def __init__(self, retry_after: str | None = None):
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = SimpleNamespace(headers=headers)


def test_format_error_exposes_only_status_and_message():
    class InternalProviderFunctionName(Exception):
        status_code = 400

        def __str__(self):
            return "The request payload is invalid"

    detail = helper._format_error(InternalProviderFunctionName())

    assert detail == "error_code=400, error_message=The request payload is invalid"
    assert "InternalProviderFunctionName" not in detail


@pytest.mark.asyncio
async def test_acompletion_always_streams_with_usage(monkeypatch):
    captured_kwargs = None

    async def fake_acompletion(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs

        async def stream():
            yield "ok"

        return stream()

    monkeypatch.setattr(helper, "acompletion", fake_acompletion)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", asyncio.Semaphore(1))

    client = LiteLLMRetryClient(max_retries=0)
    response = client.acompletion(model="test", stream=False)

    assert [chunk async for chunk in response] == ["ok"]
    assert captured_kwargs["stream"] is True
    assert captured_kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_retries_rate_limit_raised_during_iteration(monkeypatch):
    calls = 0
    sleeps: list[float] = []

    async def fake_acompletion(**kwargs):
        nonlocal calls
        calls += 1

        async def stream():
            if calls == 1:
                raise FakeRateLimitError("0")
            yield "ok"

        return stream()

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    monkeypatch.setattr(helper, "RateLimitError", FakeRateLimitError)
    monkeypatch.setattr(helper, "acompletion", fake_acompletion)
    monkeypatch.setattr(helper.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", asyncio.Semaphore(1))

    client = LiteLLMRetryClient(max_retries=1, jitter=0)
    response = client.acompletion()

    assert [chunk async for chunk in response] == ["ok"]
    assert calls == 2
    assert sleeps == [0]


@pytest.mark.asyncio
async def test_stream_propagates_non_rate_limit_errors(monkeypatch):
    async def fake_acompletion(**_kwargs):
        async def stream():
            raise ValueError("provider rejected the request")
            yield

        return stream()

    monkeypatch.setattr(helper, "acompletion", fake_acompletion)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", asyncio.Semaphore(1))

    client = LiteLLMRetryClient(max_retries=3)
    with pytest.raises(ValueError, match="provider rejected the request") as exc_info:
        await _collect(client.acompletion())

    assert str(exc_info.value) == "provider rejected the request"


@pytest.mark.asyncio
async def test_stream_holds_semaphore_until_consumption_finishes(monkeypatch):
    first_stream_started = asyncio.Event()
    release_first_stream = asyncio.Event()
    calls = 0
    active_streams = 0
    max_active_streams = 0

    async def fake_acompletion(**kwargs):
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

    monkeypatch.setattr(helper, "acompletion", fake_acompletion)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", asyncio.Semaphore(1))

    client = LiteLLMRetryClient(max_retries=0)
    first = client.acompletion()
    second = client.acompletion()

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
async def test_cancel_while_opening_completion_returns_permit(monkeypatch):
    started = asyncio.Event()
    semaphore = asyncio.Semaphore(1)

    async def fake_acompletion(**kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(helper, "acompletion", fake_acompletion)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", semaphore)
    stream = LiteLLMRetryClient().acompletion()
    task = asyncio.create_task(_collect(stream))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(semaphore.acquire(), timeout=1)
    assert semaphore.locked()
    semaphore.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["normal", "error", "timeout"])
async def test_close_returns_permit_before_sdk_cleanup(monkeypatch, cleanup):
    semaphore = asyncio.Semaphore(1)
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

    async def fake_acompletion(**kwargs):
        return SDKStream()

    monkeypatch.setattr(helper, "acompletion", fake_acompletion)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", semaphore)
    monkeypatch.setattr(helper, "STREAM_CLOSE_TIMEOUT", 0.05)
    stream = LiteLLMRetryClient(max_retries=0).acompletion()
    assert await anext(stream) == "chunk"
    assert semaphore.locked()

    close_task = asyncio.create_task(stream.aclose())
    await asyncio.wait_for(close_started.wait(), timeout=1)
    await asyncio.wait_for(semaphore.acquire(), timeout=1)
    assert semaphore.locked()
    semaphore.release()
    if cleanup == "normal":
        finish_close.set()
    await asyncio.wait_for(close_task, timeout=1)


@pytest.mark.asyncio
async def test_task_exit_and_later_stream_close_release_exactly_once(monkeypatch):
    semaphore = asyncio.Semaphore(1)
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

    async def fake_acompletion(**kwargs):
        return SDKStream()

    monkeypatch.setattr(helper, "acompletion", fake_acompletion)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", semaphore)
    stream = LiteLLMRetryClient().acompletion()

    async def consume_one_chunk():
        assert await anext(stream) == "chunk"

    await asyncio.create_task(consume_one_chunk())
    await asyncio.wait_for(closed.wait(), timeout=1)
    await stream.aclose()

    assert close_count == 1
    await asyncio.wait_for(semaphore.acquire(), timeout=1)
    assert semaphore.locked()
    semaphore.release()


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_permit_does_not_release_another_slot(monkeypatch):
    semaphore = asyncio.Semaphore(0)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", semaphore)
    stream = LiteLLMRetryClient().acompletion()
    task = asyncio.create_task(_collect(stream))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert semaphore.locked()
