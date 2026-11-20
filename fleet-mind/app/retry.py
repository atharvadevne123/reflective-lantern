"""Retry decorator with exponential backoff for transient failures."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a function on specified exceptions.

    Args:
        max_attempts: Total number of attempts before re-raising.
        base_delay: Initial sleep duration in seconds between retries.
        backoff: Multiplier applied to delay on each successive failure.
        exceptions: Exception types that trigger a retry.

    Returns:
        Decorated function with retry behaviour.

    Example::

        @retry(max_attempts=3, exceptions=(OSError,))
        def load_file(path: str) -> str:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    logger.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator
