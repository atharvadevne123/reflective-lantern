"""Retry and circuit-breaker utilities for Ticket-Oracle.

Provides a simple exponential-backoff decorator and a lightweight
in-process circuit breaker for downstream calls (DB writes, FAISS
persistence) that should not crash the prediction path.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry a function with exponential backoff.

    Args:
        max_attempts: Maximum total call attempts (including first).
        backoff_base: Base delay in seconds; doubled on each retry.
        exceptions: Exception types that trigger a retry.

    Returns:
        Decorated function.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = backoff_base
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            extra={"fn": fn.__name__, "attempts": attempt, "error": str(exc)},
                        )
                        raise
                    logger.warning(
                        "retry_attempt",
                        extra={"fn": fn.__name__, "attempt": attempt, "delay_s": delay, "error": str(exc)},
                    )
                    time.sleep(delay)
                    delay *= 2
        return wrapper  # type: ignore[return-value]
    return decorator


class CircuitBreaker:
    """Minimal in-process circuit breaker.

    Opens after `failure_threshold` consecutive failures; resets after
    `recovery_timeout` seconds of open state (half-open probe).
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._recovery_timeout:
            self._failures = 0
            self._opened_at = None
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._opened_at = time.monotonic()
            logger.error("circuit_breaker_opened", extra={"failures": self._failures})

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.is_open:
            raise RuntimeError("Circuit breaker is open; downstream call blocked.")
        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise
