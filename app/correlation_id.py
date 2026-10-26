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
    "format_correlation_id",
    "get_correlation_id",
    "is_valid_uuid",
    "new_correlation_id",
    "require_correlation_id",
    "set_correlation_id",
    "short_correlation_id",
]

_local = threading.local()


def new_correlation_id() -> str:
    """Return a fresh UUID4 string suitable as a correlation ID.

    Returns:
        36-character UUID4 string in canonical hyphenated format.
    """
    return str(uuid.uuid4())


def get_correlation_id() -> str | None:
    """Return the correlation ID for the current thread, or *None*.

    Returns:
        The currently active correlation ID string, or ``None`` when none has
        been set on this thread.
    """
    return getattr(_local, "correlation_id", None)


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current thread.

    Args:
        cid: The correlation ID string to store on thread-local storage.
    """
    _local.correlation_id = cid


def clear_correlation_id() -> None:
    """Remove the correlation ID from the current thread.

    No-ops when no ID is currently set.
    """
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


def require_correlation_id() -> str:
    """Return the current correlation ID or raise RuntimeError if unset."""
    cid = get_correlation_id()
    if cid is None:
        raise RuntimeError("No correlation ID set for this thread")
    return cid


def is_valid_uuid(value: str) -> bool:
    """Return True if *value* is a valid UUID string (any version)."""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def short_correlation_id(length: int = 8) -> str:
    """Return a shortened correlation ID (first *length* hex chars of a UUID4).

    Args:
        length: Number of hex characters to return (1-32).

    Returns:
        Lowercase hex string of the requested length.
    """
    if not (1 <= length <= 32):
        raise ValueError("length must be between 1 and 32")
    return uuid.uuid4().hex[:length]


def format_correlation_id(cid: str, prefix: str = "req-") -> str:
    """Return *cid* with a human-readable *prefix* prepended.

    Args:
        cid: Correlation ID string.
        prefix: Prefix string, default ``"req-"``.

    Returns:
        ``"{prefix}{cid}"`` string.
    """
    return f"{prefix}{cid}"
