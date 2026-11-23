"""TTL cache and request coalescing for async functions across processes."""

import hashlib
import json
from collections.abc import Callable
from decimal import Decimal
from functools import wraps
from inspect import Signature, signature
from typing import get_type_hints

from fastapi.encoders import jsonable_encoder
from loguru import logger
from pydantic import TypeAdapter
from redis.exceptions import RedisError

from . import client as connection
from .semaphore import PermitLostError, _lease

_MISS = object()
_CACHE_ERRORS = (RedisError, TimeoutError, PermitLostError)


def redis_alru_cache(*, ttl: int, scope: Callable[[object], object] | None = None, cache_none: bool = True):
    """Cache async results for ``ttl`` seconds. For methods, ``scope(self)`` replaces the instance in the key.

    Return annotations restore Python types on cache hits; untyped results must use JSON types.
    Redis errors bypass the cache. This is a TTL cache, not a process-local size-bounded LRU.
    """
    if ttl < 1:
        raise ValueError("Redis cache ttl must be positive")

    def decorate(function):
        parameters = signature(function)
        identity = (function.__module__, function.__qualname__)
        receiver = next(iter(parameters.parameters)) if scope is not None else None
        adapter = None
        if parameters.return_annotation is not Signature.empty:
            adapter = TypeAdapter(get_type_hints(function, include_extras=True)["return"])

        @wraps(function)
        async def wrapped(*args, **kwargs):
            bound = parameters.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            scope_value = scope(arguments.pop(receiver)) if scope is not None else None
            encoded_key = json.dumps([identity, scope_value, arguments], sort_keys=True, default=str, ensure_ascii=False)
            digest = hashlib.sha256(encoded_key.encode()).hexdigest()
            key = f"xpeech:cache:v2:{digest}"
            return await _cached_query(key, ttl, lambda: function(*args, **kwargs), adapter, cache_none)

        return wrapped

    return decorate


async def _cached_query(key, ttl, compute, adapter, cache_none):
    result = _MISS
    source_failed = False

    async def read(client):
        raw = await client.get(key)
        if raw is not None:
            try:
                return json.loads(raw) if adapter is None else adapter.validate_json(raw)
            except (ValueError, TypeError):
                logger.warning("Ignoring malformed Redis cache key={}", key)
        return _MISS

    try:
        async with connection.redis_client() as client:
            result = await read(client)
            if result is not _MISS:
                return result
            # 缓存锁的上限是 1；拿到锁后再查一次，之前的持有者可能已经填好缓存。
            async with _lease(client, key + ":lock", 1):
                result = await read(client)
                if result is not _MISS:
                    return result
                try:
                    result = await compute()
                except _CACHE_ERRORS:
                    source_failed = True
                    raise
                if result is not None or cache_none:
                    payload = jsonable_encoder(result, custom_encoder={Decimal: str}, sqlalchemy_safe=False)
                    await client.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl)
                return result
    except _CACHE_ERRORS:
        if source_failed:
            raise
        logger.warning("Redis cache unavailable key={}", key)
        return await compute() if result is _MISS else result
