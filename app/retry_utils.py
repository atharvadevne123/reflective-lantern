"""Retry helpers with exponential back-off and jitter."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RetryError",
    "compute_backoff_seconds",
    "retry",
    "retry_with_backoff",
]


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""


def compute_backoff_seconds(
    attempt: int,
    base: float = 1.0,
    multiplier: float = 2.0,
    cap: float = 60.0,
) -> float:
    """Return exponential back-off seconds for the given attempt (1-indexed).

    Args:
        attempt: Current attempt number, starting at 1.
        base: Starting delay in seconds.
        multiplier: Exponential multiplier per attempt.
        cap: Maximum delay returned.

    Returns:
        Capped exponential delay in seconds.
    """
    delay = base * (multiplier ** (attempt - 1))
    return min(delay, cap)


def retry(
    func: Callable[..., Any],
    *args: Any,
    attempts: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Call *func* up to *attempts* times, re-raising on the last failure.

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to *func*.
        attempts: Maximum number of invocations.
        exceptions: Exception types that trigger a retry.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Return value of the first successful call.

    Raises:
        RetryError: When all attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_exc: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            logger.debug("Attempt %d/%d failed: %s", i, attempts, exc)
    raise RetryError(f"All {attempts} attempts failed") from last_exc


def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    attempts: int = 3,
    base: float = 0.1,
    multiplier: float = 2.0,
    cap: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Like :func:`retry` but sleeps between attempts using exponential back-off.

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to *func*.
        attempts: Maximum number of invocations.
        base: Starting delay in seconds.
        multiplier: Exponential multiplier applied per attempt.
        cap: Maximum sleep delay in seconds.
        exceptions: Exception types that trigger a retry.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Return value of the first successful call.

    Raises:
        RetryError: When all attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_exc: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            if i < attempts:
                delay = compute_backoff_seconds(i, base=base, multiplier=multiplier, cap=cap)
                logger.debug("Attempt %d/%d failed; sleeping %.2fs: %s", i, attempts, delay, exc)
                time.sleep(delay)
    raise RetryError(f"All {attempts} attempts failed") from last_exc
