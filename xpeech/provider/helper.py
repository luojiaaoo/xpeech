import asyncio
import random
import time
from typing import Any

from litellm import RateLimitError, acompletion
from loguru import logger


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
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        jitter: float = 0.25,
    ):
        self.max_retries = max(0, max_retries)
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    async def acompletion(self, **kwargs: Any) -> Any:
        """Retry only recoverable rate-limit failures."""
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                result = await acompletion(**kwargs)
                elapsed = time.time() - start_time
                logger.info(
                    "LiteLLM request succeeded in {:.2f}s (attempts: {})",
                    elapsed,
                    attempt + 1,
                )
                return result
            except RateLimitError as exc:
                if attempt >= self.max_retries:
                    raise

                delay = self._retry_delay(exc, attempt)
                logger.warning(
                    "LiteLLM request was rate limited, retrying in {:.2f}s ({}/{})",
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("LiteLLM retry loop exited unexpectedly.")

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
