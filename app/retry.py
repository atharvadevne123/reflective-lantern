"""Retry utilities with exponential backoff and jitter.

This module provides decorator and helper utilities for retrying functions
on transient failures without coupling the caller to retry logic.

Key components:

- :func:`retry` — configurable decorator with exponential backoff, a jitter
  fraction to spread retries, a per-exception filter, and a ``max_delay`` cap.
- :func:`retry_on_network_error` — convenience wrapper pre-configured for
  :class:`ConnectionError`, :class:`TimeoutError`, :class:`OSError`, and
  :class:`urllib.error.URLError`.
- :func:`with_retry` — inline retry without decorator syntax; useful for
  lambdas and one-off calls.

All sleep calls go through :func:`time.sleep`, making them easy to monkeypatch
in tests without touching wall-clock time.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)


def retry(
    exceptions: Sequence[type[Exception]] = (Exception,),
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
) -> Callable[..., Any]:
    """Decorator that retries a function on failure with exponential backoff.

    Args:
        exceptions: Exception types that trigger a retry.
        max_attempts: Maximum number of total attempts.
        base_delay: Initial wait in seconds before first retry.
        max_delay: Cap on wait time between retries.
        backoff: Multiplicative factor applied to delay after each attempt.
        jitter: Fraction of delay added randomly to avoid thundering herd.

    Returns:
        Decorated function that retries on the specified exceptions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> object:
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    actual_delay = min(delay * (1 + jitter * (attempt - 1)), max_delay)
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        exc,
                        actual_delay,
                    )
                    time.sleep(actual_delay)
                    delay = min(delay * backoff, max_delay)
            logger.error(
                "All %d attempts failed for %s",
                max_attempts,
                func.__qualname__,
            )
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


def retry_on_network_error(max_attempts: int = 3, base_delay: float = 2.0) -> Callable[..., Any]:
    """Convenience wrapper for retrying on common network-related exceptions.

    Args:
        max_attempts: Maximum number of total attempts.
        base_delay: Initial wait in seconds before first retry.

    Returns:
        Decorator configured for network errors.
    """
    import urllib.error

    return retry(
        exceptions=(ConnectionError, TimeoutError, urllib.error.URLError, OSError),
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff=2.0,
    )


def with_retry(func: Callable, *args, max_attempts: int = 3, **kwargs) -> object:
    """Call *func* with retry logic applied inline (no decorator needed).

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to func.
        max_attempts: Maximum number of total attempts.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        Whatever func returns on success.

    Raises:
        Exception: Last exception raised after all attempts are exhausted.
    """
    wrapped = retry(max_attempts=max_attempts)(func)
    return wrapped(*args, **kwargs)


__all__ = ["retry", "retry_on_network_error", "with_retry"]
