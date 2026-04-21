from __future__ import annotations

import asyncio
import logging
from random import uniform
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryConfig:
    def __init__(
        self,
        attempts: int = 3,
        base_delay_sec: float = 0.5,
        max_delay_sec: float = 10.0,
        jitter: bool = True,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.attempts = attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    config: RetryConfig,
    operation_name: str = "operation",
) -> T:
    last_exc: Exception | None = None
    attempts = max(1, config.attempts)
    for attempt in range(attempts):
        try:
            return await fn()
        except config.retryable_exceptions as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status is not None and status not in RETRYABLE_STATUS_CODES:
                raise

            if attempt == attempts - 1:
                break

            delay = min(config.base_delay_sec * (2**attempt), config.max_delay_sec)
            if config.jitter:
                delay += uniform(0, config.base_delay_sec)

            logger.warning(
                "Retrying %s after error: %s (attempt %d/%d, delay=%.2fs)",
                operation_name,
                exc,
                attempt + 1,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Retry failed for {operation_name}")
