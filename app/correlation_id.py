"""Request correlation-ID utilities for distributed tracing.

Provides thread-local storage and context-manager helpers so every log
line emitted during a request automatically carries a trace identifier.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager

__all__ = [
    "clear_correlation_id",
    "correlation_context",
    "get_correlation_id",
    "new_correlation_id",
    "set_correlation_id",
]

_local = threading.local()


def new_correlation_id() -> str:
    """Return a fresh UUID4 string suitable as a correlation ID."""
    return str(uuid.uuid4())


def get_correlation_id() -> str | None:
    """Return the correlation ID for the current thread, or *None*."""
    return getattr(_local, "correlation_id", None)


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current thread."""
    _local.correlation_id = cid


def clear_correlation_id() -> None:
    """Remove the correlation ID from the current thread."""
    if hasattr(_local, "correlation_id"):
        del _local.correlation_id


@contextmanager
def correlation_context(cid: str | None = None) -> Generator[str, None, None]:
    """Context manager that sets a correlation ID and restores the previous one.

    Args:
        cid: Correlation ID to use. A new UUID is generated when *None*.

    Yields:
        The active correlation ID string.
    """
    previous = get_correlation_id()
    active = cid or new_correlation_id()
    set_correlation_id(active)
    try:
        yield active
    finally:
        if previous is None:
            clear_correlation_id()
        else:
            set_correlation_id(previous)
