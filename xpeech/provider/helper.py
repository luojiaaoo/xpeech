import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Any

from litellm import RateLimitError, acompletion
from loguru import logger

from ..config.settings import settings

LLM_PARALLEL_SEMAPHORE = asyncio.Semaphore(settings.llm.parallel)


class LiteLLMRetryClient:
    """Call LiteLLM with retries for rate-limit responses only.

    Args:
        max_retries: Number of retries after the first failed request.
        initial_delay: Initial backoff delay in seconds.
        max_delay: Maximum delay between retry attempts in seconds.
        jitter: Extra random delay in seconds to avoid retry bursts.
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

    async def acompletion(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream a completion while retaining concurrency and retry guards."""
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {"include_usage": True})
        start_time = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                async with LLM_PARALLEL_SEMAPHORE:
                    response = await acompletion(**kwargs)
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
                    raise

                delay = self._retry_delay(exc, attempt)
                logger.warning(
                    "LiteLLM stream was rate limited, retrying in {:.2f}s ({}/{})",
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                await asyncio.sleep(delay)

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
