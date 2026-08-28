"""Retry utilities with exponential backoff and jitter."""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Sequence, Type

logger = logging.getLogger(__name__)


def retry(
    exceptions: Sequence[Type[Exception]] = (Exception,),
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
) -> Callable:
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
    def decorator(func: Callable) -> Callable:
        """Wrap *func* with retry logic using the captured parameters."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[return]
            """Execute *func* with exponential backoff retries on failure."""
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


def retry_on_network_error(max_attempts: int = 3, base_delay: float = 2.0) -> Callable:
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


__all__ = ["retry", "retry_on_network_error"]
