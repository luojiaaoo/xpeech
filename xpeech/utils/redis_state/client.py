"""Share an asynchronous Redis connection pool within each service event loop."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis import asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

_pools: dict[asyncio.AbstractEventLoop, aioredis.ConnectionPool] = {}


def _new_pool() -> aioredis.ConnectionPool:
    from ...config.settings import settings

    connection = settings.redis
    return aioredis.ConnectionPool(
        connection_class=aioredis.SSLConnection if connection.ssl else aioredis.Connection,
        host=connection.host, port=connection.port, db=connection.db,
        username=connection.username, password=connection.password.get_secret_value() or None,
        socket_connect_timeout=min(connection.timeout, 10),
        socket_timeout=min(connection.timeout, 10), decode_responses=True,
        retry=Retry(NoBackoff(), 0),
    )


@asynccontextmanager
async def redis_pool() -> AsyncIterator[None]:
    """Keep a shared pool open for the service lifespan, without connecting eagerly."""
    loop = asyncio.get_running_loop()
    pool = _new_pool()
    _pools[loop] = pool
    try:
        yield
    finally:
        del _pools[loop]
        await pool.aclose()


def redis_client() -> aioredis.Redis:
    pool = _pools.get(asyncio.get_running_loop())
    if pool is None:
        return aioredis.Redis.from_pool(_new_pool())
    return aioredis.Redis(connection_pool=pool)
