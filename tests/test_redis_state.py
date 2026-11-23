import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError
from typing_extensions import TypedDict

from xpeech.config.settings import LLMConfig, settings
from xpeech.utils import redis_state as state
from xpeech.utils.redis_state import client as redis_connection
from xpeech.utils.redis_state.client import redis_client as real_redis_client


@pytest.mark.asyncio
async def test_clients_share_pool_until_lifespan_ends(monkeypatch):
    async with state.redis_pool():
        async with real_redis_client() as first, real_redis_client() as second:
            pool = first.connection_pool
            assert second.connection_pool is pool
            close = AsyncMock(wraps=pool.aclose)
            monkeypatch.setattr(pool, "aclose", close)
        close.assert_not_awaited()

    close.assert_awaited_once()
    async with real_redis_client() as standalone:
        assert standalone.connection_pool is not pool


@pytest.mark.asyncio
async def test_separate_clients_share_limit_and_reclaim_expired_permits():
    async with redis_connection.redis_client() as first:
        await first.zadd("permits", {"crashed": 0})
        entered = asyncio.Event()

        async def waiting():
            async with state.redis_semaphore(1, key="permits"):
                entered.set()

        async with state.redis_semaphore(1, key="permits"):
            task = asyncio.create_task(waiting())
            await asyncio.sleep(0.05)
            assert not entered.is_set()
            assert await first.zcard("permits") == 1
        await asyncio.wait_for(task, 1)
        assert entered.is_set()
        assert await first.zcard("permits") == 0


@pytest.mark.asyncio
async def test_lease_is_renewed_and_cancelled_holder_releases():
    async with redis_connection.redis_client() as client:
        entered = asyncio.Event()

        async def holder():
            async with state.redis_semaphore(1, key="renew", lease_ms=400):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        await asyncio.sleep(0.65)
        assert await client.zcard("renew") == 1
        assert await client.pttl("renew") > 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await client.zcard("renew") == 0


@pytest.mark.asyncio
async def test_lost_lease_stops_holder():
    async with redis_connection.redis_client() as client:
        entered = asyncio.Event()

        async def holder():
            async with state.redis_semaphore(1, key="lost", lease_ms=400):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        await client.delete("lost")
        with pytest.raises(state.PermitLostError):
            await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_cancelled_waiter_cannot_remove_holders_permit():
    async with redis_connection.redis_client() as client, state.redis_semaphore(1, key="waiting"):
        async def waiter():
            async with state.redis_semaphore(1, key="waiting"):
                pytest.fail("Waiter acquired an occupied permit")

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await client.zcard("waiting") == 1


@pytest.mark.asyncio
async def test_shared_cache_singleflight_ttl_types_and_null():
    class CachedValue(TypedDict):
        time: datetime
        n: Decimal

    value = {"time": datetime(2026, 11, 21, tzinfo=UTC), "n": Decimal("12.3456789012345678901234")}
    compute = AsyncMock(return_value=value)

    @state.redis_alru_cache(ttl=5)
    async def cached(limit=20) -> CachedValue:
        return await compute(limit)

    results = await asyncio.gather(cached(), cached(20), cached(limit=20), cached())
    assert all(result == value for result in results)
    compute.assert_awaited_once()
    async with redis_connection.redis_client() as client:
        (key,) = await client.keys("xpeech:cache:v2:*")
        assert json.loads(await client.get(key)) == {"time": value["time"].isoformat(), "n": str(value["n"])}
        assert 0 < await client.ttl(key) <= 5
        await client.pexpire(key, 1)
    await asyncio.sleep(0.02)
    await cached()
    assert compute.await_count == 2

    null = AsyncMock(return_value=None)

    @state.redis_alru_cache(ttl=5)
    async def cached_null():
        return await null()

    assert await cached_null() is None
    assert await cached_null() is None
    null.assert_awaited_once()


@pytest.mark.asyncio
async def test_failures_are_not_cached():
    compute = AsyncMock(side_effect=[TimeoutError("source timed out"), None, "ok"])

    @state.redis_alru_cache(ttl=3600, cache_none=False)
    async def cached():
        return await compute()

    with pytest.raises(TimeoutError, match="source timed out"):
        await cached()
    assert await cached() is None
    assert await cached() == "ok"
    assert await cached() == "ok"
    assert compute.await_count == 3


@pytest.mark.asyncio
async def test_cache_outage_falls_back_but_model_outage_rejects(monkeypatch, make_openai_sdk_client):
    from xpeech.provider import openai_stream

    @asynccontextmanager
    async def unavailable():
        raise ConnectionError("offline")
        yield

    monkeypatch.setattr(redis_connection, "redis_client", unavailable)
    compute = AsyncMock(return_value="source")

    @state.redis_alru_cache(ttl=5)
    async def cached():
        return await compute()

    assert await cached() == "source"
    compute.assert_awaited_once()
    completion = AsyncMock()
    client = openai_stream.OpenAIStreamClient(make_openai_sdk_client(completion))
    with pytest.raises(ConnectionError):
        await anext(client.create())
    completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_write_failure_does_not_repeat_source_query(monkeypatch):
    original_client = redis_connection.redis_client

    def client():
        result = original_client()
        result.set = AsyncMock(side_effect=ConnectionError("write failed"))
        return result

    monkeypatch.setattr(redis_connection, "redis_client", client)
    compute = AsyncMock(return_value="already computed")

    @state.redis_alru_cache(ttl=5)
    async def cached():
        return await compute()

    assert await cached() == "already computed"
    compute.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_limit_is_shared_by_stream_clients(monkeypatch, make_openai_sdk_client):
    from xpeech.provider import openai_stream

    monkeypatch.setattr(settings.llm, "parallel", 1)
    started = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def completion(**kwargs):
        nonlocal calls
        calls += 1

        async def stream():
            started.set()
            await finish.wait()
            yield "ok"

        return stream()

    async def consume():
        client = openai_stream.OpenAIStreamClient(make_openai_sdk_client(completion))
        return [part async for part in client.create()]

    first = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 1)
    second = asyncio.create_task(consume())
    try:
        await asyncio.sleep(0.05)
        assert calls == 1
    finally:
        finish.set()
        assert await asyncio.wait_for(asyncio.gather(first, second), 2) == [["ok"], ["ok"]]
    assert calls == 2


@pytest.mark.asyncio
async def test_statistics_share_cache_across_repository_instances():
    from xpeech.agent.record import ConversationRecord
    from xpeech.agent.server.routes.statistics import _CachedRecord, _StatisticsValues, statistics_cache

    record = ConversationRecord(
        id=1, user_id="alice", session_id="session", request_id="request", sender_name="Alice",
        user_question="Q", model_response="A", input_tokens=12, output_tokens=4,
        model_call_count=1, duration_s=1, channel="web_client",
    )
    summary_values = {"input_tokens": Decimal(16), "data_as_of": record.created_at}
    compute = AsyncMock(return_value=([record], summary_values))

    class Repository:
        def __init__(self, database="database-a"):
            self._engine = SimpleNamespace(url=SimpleNamespace(render_as_string=lambda **kwargs: database))

        @statistics_cache
        async def records(self, *, limit=20) -> tuple[list[_CachedRecord], _StatisticsValues]:
            return await compute()

    first, second = Repository(), Repository()
    await first.records()
    result, summary = await second.records(limit=20)
    assert isinstance(result[0], ConversationRecord)
    assert result[0].created_at == record.created_at
    assert summary == summary_values
    assert isinstance(summary["input_tokens"], Decimal)
    compute.assert_awaited_once()
    await second.records(limit=10)
    assert compute.await_count == 2
    await Repository("database-b").records()
    assert compute.await_count == 3


@pytest.mark.asyncio
async def test_extra_info_shared_across_config_instances_and_configuration_changes(monkeypatch):
    from importlib import import_module

    from xpeech.config.settings import ExtraUserInfoConfig

    module = import_module("xpeech.config.settings")
    compute = AsyncMock(return_value='{"org":"company/team","name":"Alice"}')
    monkeypatch.setattr(module, "execute_command_template", compute)
    config = {"command_extra_user_info_template": "lookup $user_id", "user_organization_claim": ".org", "username_claim": ".name"}
    assert await ExtraUserInfoConfig(**config).resolve_sender_name("alice") == "Alice"
    assert await ExtraUserInfoConfig(**config).resolve_user_organization("alice") == "company/team"
    compute.assert_awaited_once()
    await ExtraUserInfoConfig(**config).resolve_sender_name("bob")
    await ExtraUserInfoConfig(**{**config, "username_claim": ".org"}).resolve_sender_name("alice")
    assert compute.await_count == 3


def test_single_api_key_and_positive_cluster_limit():
    values = settings.llm.model_dump()
    config = LLMConfig(**{**values, "api_key": " single "})
    assert [config.api_key for _ in range(3)] == ["single"] * 3
    for key in ("", "a,b", "a,"):
        with pytest.raises(ValueError):
            LLMConfig(**{**values, "api_key": key})
    with pytest.raises(ValueError):
        LLMConfig(**{**values, "parallel": 0})
