"""A distributed semaphore backed by renewable Redis leases."""

import asyncio
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import anyio
from loguru import logger
from redis.asyncio import Redis
from redis.exceptions import RedisError

from . import client as connection


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
async def redis_semaphore(value: int, *, key: str, lease_ms: int = 60000) -> AsyncIterator[None]:
    """Acquire on entering ``async with`` and release on exit, across processes sharing ``key``."""
    if value < 1 or lease_ms < 1:
        raise ValueError("Redis semaphore value and lease_ms must be positive")
    async with connection.redis_client() as client, _lease(client, key, value, lease_ms=lease_ms):
        yield


@asynccontextmanager
async def _lease(client: Redis, key: str, limit: int, *, lease_ms: int = 60000) -> AsyncIterator[None]:
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
                # 无法确认名额仍归自己，就停止使用名额的任务。
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
