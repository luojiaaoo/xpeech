import asyncio
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from litellm import RateLimitError, acompletion
from loguru import logger
from openai import APIStatusError

from ..config.settings import settings

LLM_PARALLEL_SEMAPHORE = asyncio.Semaphore(settings.llm.parallel)
STREAM_CLOSE_TIMEOUT = 5.0


async def _close_stream(response: Any) -> None:
    """限时清理 HTTP 连接，清理期间不占用模型并发名额。"""
    close = getattr(response, "aclose", None)
    if close is None:
        return
    with anyio.move_on_after(STREAM_CLOSE_TIMEOUT, shield=True) as close_scope:
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - 清理异常不能覆盖原始的流异常。
            logger.opt(exception=exc).warning("Failed to close LiteLLM stream")
    if close_scope.cancel_called:
        logger.warning("Timed out closing LiteLLM stream")


@asynccontextmanager
async def _completion_stream(**kwargs: Any) -> AsyncIterator[Any]:
    """通过调用方的 aclosing 关闭链归还模型名额，再限时清理上游连接。"""
    semaphore = LLM_PARALLEL_SEMAPHORE
    await semaphore.acquire()
    response = None
    try:
        response = await acompletion(**kwargs)
        yield response
    finally:
        semaphore.release()
        if response is not None:
            await _close_stream(response)


def _format_error(exc: APIStatusError) -> str:
    return f"error_code={exc.status_code}, error_message={exc}"


class LiteLLMRetryClient:
    """调用 LiteLLM，仅在收到限流响应时重试。

    参数：
        max_retries: 首次请求失败后的最大重试次数。
        initial_delay: 初始退避时间，单位为秒。
        max_delay: 重试间隔的上限，单位为秒。
        jitter: 额外的随机延迟范围，单位为秒，用于分散重试请求。
    """

    def __init__(
        self,
        max_retries: int = 8,
        initial_delay: float = 2.0,
        max_delay: float = 12.0,
        jitter: float = 0.5,
    ):
        self.max_retries = max(0, max_retries)
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    async def acompletion(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """流式获取模型响应，并保留并发限制和限流重试机制。"""
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {"include_usage": True})
        start_time = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                async with _completion_stream(**kwargs) as response:
                    async for chunk in response:
                        yield chunk

                elapsed = time.time() - start_time
                logger.info(
                    "LiteLLM stream succeeded in {:.2f}s (attempts: {})",
                    elapsed,
                    attempt + 1,
                )
                return
            except RateLimitError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(_format_error(exc)) from exc

                delay = self._retry_delay(exc, attempt)
                logger.warning(
                    "LiteLLM stream was rate limited, retrying in {:.2f}s ({}/{})",
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                await asyncio.sleep(delay)
            except APIStatusError as exc:
                raise RuntimeError(_format_error(exc)) from exc
            except Exception:
                raise

        # 理论上不可达
        raise RuntimeError("LiteLLM streaming retry loop exited unexpectedly.")

    def _retry_delay(self, exc: RateLimitError, attempt: int) -> float:
        retry_after = exc.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0), self.max_delay)
            except ValueError:
                pass

        delay = min(self.initial_delay * (2**attempt), self.max_delay)
        if self.jitter <= 0:
            return delay

        return min(delay + random.uniform(0, self.jitter), self.max_delay)
