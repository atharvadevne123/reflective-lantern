"""In-memory TTL cache for prediction results."""

from __future__ import annotations

import time
from typing import Any

_store: dict[str, tuple[Any, float]] = {}
DEFAULT_TTL = 60.0


def set_cache(key: str, value: Any, ttl: float = DEFAULT_TTL) -> None:
    """Store *value* under *key* with a TTL of *ttl* seconds."""
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
    """Remove the cache entry for *key* if it exists."""
    _store.pop(key, None)


def clear_all() -> None:
    """Evict all entries from the cache."""
    _store.clear()


def cache_size() -> int:
    """Return the number of non-expired entries currently in the cache."""
    now = time.monotonic()
    return sum(1 for _, (_, exp) in _store.items() if now <= exp)
