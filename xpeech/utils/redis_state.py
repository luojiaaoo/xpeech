"""Shared Redis state. Connections are scoped to an operation/event loop."""

import asyncio
import hashlib
import json
import random
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

import anyio
from loguru import logger
from pydantic import BaseModel
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

if TYPE_CHECKING:
    from ..config.settings import RedisConfig


def redis_client(connection: "RedisConfig | None" = None) -> Redis:
    # Redis 用 key 存取数据；连到同一个 Redis 服务和同一个 db 的进程共享这些 key。
    if connection is None:
        from ..config.settings import settings

        connection = settings.redis
    return Redis(
        host=connection.host, port=connection.port, db=connection.db,
        username=connection.username, password=connection.password.get_secret_value() or None,
        ssl=connection.ssl, socket_connect_timeout=min(connection.timeout, 10),
        socket_timeout=min(connection.timeout, 10), decode_responses=True,
        retry=Retry(NoBackoff(), 0),
    )


def state_key(namespace: str, values: object) -> str:
    # 用命名空间区分用途，用参数哈希生成固定长度的 key，不直接把参数拼进去。
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()
    return f"xpeech:{namespace}:v1:{digest}"


# ZSET（有序集合）在这里记录已占用的名额：成员是持有者的 UUID，分数是租约到期时间（毫秒）。
# KEYS[1] 是集合的 key；申请时 ARGV 依次是 UUID、并发上限、租约时长。
# Lua 在 Redis 内原子执行：用 Redis 自己的时钟清理到期成员，再用 ZCARD 计数，没满才 ZADD。
# PEXPIRE 设置整个 key 的过期时间，不是给单个成员设置；单个名额何时过期由分数决定。
_ACQUIRE = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then return 0 end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[3]), ARGV[1])
redis.call('PEXPIRE', KEYS[1], ARGV[3])
return 1
"""

# ZSCORE 取出指定 UUID 的到期时间；只给仍有效的成员续租，不能把过期名额重新加回来。
_RENEW = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
local expiry = redis.call('ZSCORE', KEYS[1], ARGV[1])
if not expiry or tonumber(expiry) <= now then return 0 end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 1
"""


class PermitLostError(RuntimeError):
    """A running operation can no longer prove it owns a Redis permit."""


@asynccontextmanager
async def redis_permit(client: Redis, key: str, limit: int, *, lease_ms: int = 60000):
    """Hold a renewable permit; cancel the holder if ownership becomes uncertain."""
    permit = uuid4().hex
    owner = asyncio.current_task()
    lost = False

    async def renew():
        nonlocal lost
        try:
            while True:
                # 每过四分之一租期续租（默认 15 秒）；默认租约 60 秒不是模型调用的时限。
                await asyncio.sleep(lease_ms / 4000)
                async with asyncio.timeout(lease_ms / 6000):
                    if not await client.eval(_RENEW, 1, key, permit, lease_ms):
                        raise RuntimeError("Redis permit expired")
        except (RedisError, TimeoutError, RuntimeError):
            # 无法确认名额仍归自己，就停止使用名额的任务，而不是继续调用模型。
            lost = True
            logger.error("Redis permit renewal failed; stopping holder key={}", key)
            owner.cancel()

    task = None
    try:
        while True:
            async with asyncio.timeout(lease_ms / 6000):
                acquired = await client.eval(_ACQUIRE, 1, key, permit, limit, lease_ms)
            if acquired:
                break
            # 满额时异步等 100～300 毫秒再抢；不阻塞事件循环，也不保证先到先得。
            await asyncio.sleep(random.uniform(0.1, 0.3))
        task = asyncio.create_task(renew())
        try:
            # 只有拿到名额才进入调用方的 async with 块；退出后走 finally 清理。
            yield
            if lost:
                raise PermitLostError("Redis permit lost; operation stopped")
        except asyncio.CancelledError:
            if lost:
                owner.uncancel()
                raise PermitLostError("Redis permit lost; operation stopped") from None
            raise
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        # ZREM 只删除本次申请的 UUID，申请响应丢失时也可以尝试清理；进程崩溃则靠租约到期回收。
        # 在取消时也尽量完成清理，但最多给 Redis 5 秒。
        with anyio.CancelScope(shield=True):
            try:
                async with asyncio.timeout(5):
                    await client.zrem(key, permit)
            except (RedisError, TimeoutError):
                logger.warning("Redis permit release failed; waiting for lease expiry key={}", key)


def _encode(value):
    # JSON 没有 datetime 和 Decimal 类型，写缓存时加上类型标签，读取时才能还原。
    if isinstance(value, datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        return ["dict", {key: _encode(item) for key, item in value.items()}]
    if isinstance(value, (tuple, list)):
        return ["list", [_encode(item) for item in value]]
    return ["value", value]


def _decode(value):
    tag, data = value
    if tag == "datetime":
        return datetime.fromisoformat(data)
    if tag == "decimal":
        return Decimal(data)
    if tag == "dict":
        return {key: _decode(item) for key, item in data.items()}
    if tag == "list":
        return [_decode(item) for item in data]
    if tag == "value":
        return data
    raise ValueError("Invalid Redis cache value")


async def cached_query(key, ttl, compute, *, cache_none=True):
    """Share successful JSON results and coalesce misses across services."""
    computed = False
    result = None

    async def read(client):
        raw = await client.get(key)
        if raw is None:
            return False, None
        try:
            return True, _decode(json.loads(raw))
        except (ValueError, TypeError, KeyError):
            logger.warning("Ignoring malformed Redis cache key={}", key)
            return False, None

    try:
        async with redis_client() as client:
            found, result = await read(client)
            if found:
                return result
            # 缓存锁的上限是 1；抢到锁后再读一次，之前的持有者可能已经写好缓存。
            async with redis_permit(client, key + ":lock", 1):
                found, result = await read(client)
                if found:
                    return result
                result = await compute()
                computed = True
                if result is not None or cache_none:
                    # ex 给缓存 key 设置过期时间，单位是秒；上面的 PEXPIRE 用的是毫秒。
                    await client.set(key, json.dumps(_encode(result), ensure_ascii=False), ex=ttl)
                return result
    except (RedisError, TimeoutError, PermitLostError):
        # Redis 故障时缓存可以绕过；如果已经查到结果，就不再重复查询。
        logger.warning("Redis cache unavailable; querying source key={}", key)
        return result if computed else await compute()
