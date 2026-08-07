"""In-memory TTL cache for prediction results."""

from __future__ import annotations

import time
from typing import Any

_store: dict[str, tuple[Any, float]] = {}
DEFAULT_TTL = 60.0


def set_cache(key: str, value: Any, ttl: float = DEFAULT_TTL) -> None:
    _store[key] = (value, time.monotonic() + ttl)


def get_cache(key: str) -> Any | None:
    entry = _store.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.monotonic() > expires_at:
        del _store[key]
        return None
    return value


def invalidate(key: str) -> None:
    _store.pop(key, None)


def clear_all() -> None:
    _store.clear()


def cache_size() -> int:
    now = time.monotonic()
    return sum(1 for _, (_, exp) in _store.items() if now <= exp)
