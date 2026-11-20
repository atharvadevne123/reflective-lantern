"""In-memory TTL cache for prediction results."""

from __future__ import annotations

import time
from typing import Any

_store: dict[str, tuple[Any, float]] = {}
DEFAULT_TTL = 60.0


def set_cache(key: str, value: Any, ttl: float = DEFAULT_TTL) -> None:
    """Store *value* under *key* with a time-to-live of *ttl* seconds."""
    _store[key] = (value, time.monotonic() + ttl)


def get_cache(key: str) -> Any | None:
    """Return the cached value for *key*, or None if missing or expired."""
    entry = _store.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.monotonic() > expires_at:
        del _store[key]
        return None
    return value


def invalidate(key: str) -> None:
    """Remove *key* from the cache (no-op if absent)."""
    _store.pop(key, None)


def clear_all() -> None:
    """Evict every entry from the cache."""
    _store.clear()


def cache_size() -> int:
    """Return the number of non-expired entries currently in the cache."""
    now = time.monotonic()
    return sum(1 for _, (_, exp) in _store.items() if now <= exp)


def evict_expired() -> int:
    """Remove all expired entries and return the count removed."""
    now = time.monotonic()
    expired_keys = [k for k, (_, exp) in _store.items() if now > exp]
    for k in expired_keys:
        del _store[k]
    return len(expired_keys)


def get_with_default(key: str, default: Any = None) -> Any:
    """Return the cached value for *key*, or *default* if missing or expired.

    Args:
        key: Cache key to look up.
        default: Value to return when the key is absent or expired.

    Returns:
        Cached value or *default*.
    """
    result = get_cache(key)
    return result if result is not None else default


def keys() -> list[str]:
    """Return the keys of all non-expired cache entries."""
    now = time.monotonic()
    return [k for k, (_, exp) in _store.items() if now <= exp]
