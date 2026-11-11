import asyncio
import random
import time
from collections.abc import AsyncGenerator
from typing import Any

import anyio
from litellm import RateLimitError, acompletion
from loguru import logger
from openai import APIStatusError

from ..config.settings import settings

LLM_PARALLEL_SEMAPHORE = asyncio.Semaphore(settings.llm.parallel)
STREAM_CLOSE_TIMEOUT = 5.0
_STREAM_CLOSE_TASKS: set[asyncio.Task[None]] = set()


async def _close_stream(response: Any) -> None:
    """最多等待指定时间清理连接，不受 SDK 内部 AnyIO shield 影响。"""
    close = getattr(response, "aclose", None)
    if close is None:
        return

    async def close_response() -> None:
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - 清理异常不能覆盖原始的流异常。
            logger.opt(exception=exc).warning("Failed to close LiteLLM stream")

    task = asyncio.create_task(close_response())
    # 超时后 SDK 可能吞掉取消，保留任务引用，直至其真正结束。
    _STREAM_CLOSE_TASKS.add(task)
    task.add_done_callback(_STREAM_CLOSE_TASKS.discard)
    with anyio.CancelScope(shield=True):
        try:
            done, _ = await asyncio.wait({task}, timeout=STREAM_CLOSE_TIMEOUT)
            if not done:
                logger.warning("Timed out closing LiteLLM stream")
        finally:
            if not task.done():
                task.cancel()


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
                semaphore = LLM_PARALLEL_SEMAPHORE
                await semaphore.acquire()
                response = None
                try:
                    response = await acompletion(**kwargs)
                    async for chunk in response:
                        yield chunk
                finally:
                    semaphore.release()
                    if response is not None:
                        await _close_stream(response)

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
