import asyncio
from types import SimpleNamespace

import pytest

from xpeech.provider import helper
from xpeech.provider.helper import LiteLLMRetryClient


class FakeRateLimitError(Exception):
    def __init__(self, retry_after: str | None = None):
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = SimpleNamespace(headers=headers)


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
