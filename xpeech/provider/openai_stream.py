import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import anyio
from loguru import logger
from openai import APIStatusError, AsyncOpenAI

from ..config.settings import settings
from ..utils.redis_state import redis_semaphore

STREAM_CLOSE_TIMEOUT = 5.0
_STREAM_CLOSE_TASKS: set[asyncio.Task[None]] = set()


async def _close_stream(response: Any) -> None:
    """最多等待指定时间清理连接，不受 SDK 内部 AnyIO shield 影响。"""
    close = getattr(response, "close", None) or getattr(response, "aclose", None)
    if close is None:
        return

    async def close_response() -> None:
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - 清理异常不能覆盖原始的流异常。
            logger.opt(exception=exc).warning("Failed to close OpenAI stream")

    task = asyncio.create_task(close_response())
    # 超时后 SDK 可能吞掉取消，保留任务引用，直至其真正结束。
    _STREAM_CLOSE_TASKS.add(task)
    task.add_done_callback(_STREAM_CLOSE_TASKS.discard)
    with anyio.CancelScope(shield=True):
        try:
            done, _ = await asyncio.wait({task}, timeout=STREAM_CLOSE_TIMEOUT)
            if not done:
                logger.warning("Timed out closing OpenAI stream")
        finally:
            if not task.done():
                task.cancel()


def _format_error(exc: APIStatusError) -> str:
    return f"error_code={exc.status_code}, error_message={exc}"


class OpenAIStreamClient:
    """在持有 Redis 并发名额期间创建并消费 OpenAI 流式响应。"""

    def __init__(self, client: AsyncOpenAI):
        self._client = client

    async def create(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """流式获取模型响应；请求建立前的重试由 OpenAI SDK 负责。"""
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {"include_usage": True})
        start_time = time.time()
        response = None
        try:
            async with redis_semaphore(settings.llm.parallel, key="xpeech:llm:permits:v1"):
                response = await self._client.chat.completions.create(**kwargs)
                async for chunk in response:
                    yield chunk
        except APIStatusError as exc:
            raise RuntimeError(_format_error(exc)) from exc
        finally:
            if response is not None:
                await _close_stream(response)

        elapsed = time.time() - start_time
        logger.info("OpenAI stream succeeded in {:.2f}s", elapsed)
