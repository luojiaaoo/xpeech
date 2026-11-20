import asyncio
import secrets
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from loguru import logger
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError, TimeoutError as RedisTimeoutError

from .....config.settings import RedisConfig
from ..dao import User
from ..models import (
    INJECT_PROMPT_EXPIRES_IN,
    INJECT_PROMPT_KEY_PREFIX,
    INJECT_PROMPT_STATE_MAX_LENGTH,
    INJECT_PROMPT_STATE_MIN_LENGTH,
    INJECT_PROMPT_STATE_PATTERN,
    InjectPromptCreateBody,
)


def _prompt_redis(connection: RedisConfig) -> Redis:
    return Redis(
        host=connection.host,
        port=connection.port,
        db=connection.db,
        username=connection.username,
        password=connection.password.get_secret_value() or None,
        ssl=connection.ssl,
        socket_connect_timeout=connection.timeout,
        socket_timeout=connection.timeout,
        decode_responses=True,
        # One-shot operations cannot safely be retried after losing the response.
        retry=Retry(NoBackoff(), 0),
    )


async def resolve_injected_prompt(redis_config: RedisConfig, state: str) -> str:
    try:
        async with asyncio.timeout(redis_config.timeout):
            async with _prompt_redis(redis_config) as client:
                prompt = await client.getdel(f"{INJECT_PROMPT_KEY_PREFIX}{state}")
    except (TimeoutError, RedisTimeoutError) as error:
        logger.warning("Redis prompt lookup timed out")
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "提示词读取超时") from error
    except (RedisError, UnicodeDecodeError) as error:
        logger.warning("Redis prompt lookup failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "提示词读取失败") from error
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "未找到对应的提示词")
    return prompt


async def store_injected_prompt(redis_config: RedisConfig, body: InjectPromptCreateBody) -> str:
    try:
        async with asyncio.timeout(redis_config.timeout):
            async with _prompt_redis(redis_config) as client:
                for _ in range(3):
                    state = secrets.token_urlsafe(32)
                    stored = await client.set(
                        f"{INJECT_PROMPT_KEY_PREFIX}{state}",
                        body.user_prefix,
                        ex=INJECT_PROMPT_EXPIRES_IN,
                        nx=True,
                    )
                    if stored:
                        return state
    except (TimeoutError, RedisTimeoutError) as error:
        logger.warning("Redis prompt write timed out")
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "提示词写入超时") from error
    except RedisError as error:
        logger.warning("Redis prompt write failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "提示词写入失败") from error
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "暂时无法生成可用的 state，请重试")


def create_inject_prompt_router(
    redis_config: RedisConfig,
    current_user_dependency: Callable[..., Awaitable[User] | User],
) -> APIRouter:
    router = APIRouter(prefix="/api/auth")
    CurrentUser = Annotated[User, Depends(current_user_dependency)]

    @router.post("/inject-prompt", status_code=status.HTTP_201_CREATED)
    async def create_inject_prompt(body: InjectPromptCreateBody, response: Response):
        state = await store_injected_prompt(redis_config, body)
        response.headers["Cache-Control"] = "no-store"
        return {
            "state": state,
            "expires_in": INJECT_PROMPT_EXPIRES_IN,
            "login_url": f"/?state={state}",
        }

    @router.get("/inject-prompt")
    async def inject_prompt(
        state_value: Annotated[
            str,
            Query(
                alias="state",
                min_length=INJECT_PROMPT_STATE_MIN_LENGTH,
                max_length=INJECT_PROMPT_STATE_MAX_LENGTH,
                pattern=INJECT_PROMPT_STATE_PATTERN,
            ),
        ],
        response: Response,
        user: CurrentUser,
    ):
        del user
        response.headers["Cache-Control"] = "no-store"
        return {
            "user_prefix": await resolve_injected_prompt(
                redis_config,
                state_value,
            )
        }

    return router
