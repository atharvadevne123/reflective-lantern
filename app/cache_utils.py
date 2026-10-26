"""Simple TTL-based in-memory cache utilities."""

from __future__ import annotations

import time
from typing import Any

__all__ = [
    "TTLCache",
    "memoize_ttl",
]


class TTLCache:
    """Thread-unsafe TTL cache mapping string keys to arbitrary values.

    Expired entries are evicted lazily on access.

    Args:
        ttl: Time-to-live in seconds for each entry.
        max_size: Maximum number of entries before oldest is evicted (0 = unlimited).
    """

    def __init__(self, ttl: float = 60.0, max_size: int = 0) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        self._ttl = ttl
        self._max_size = max_size
        self._store: dict[str, tuple[Any, float]] = {}

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* with a fresh TTL."""
        if self._max_size and len(self._store) >= self._max_size and key not in self._store:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, time.monotonic() + self._ttl)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* if present and not expired, else *default*."""
        entry = self._store.get(key)
        if entry is None:
            return default
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return default
        return value

    def delete(self, key: str) -> bool:
        """Remove *key*; return True if it existed."""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def __contains__(self, key: str) -> bool:
        """Return True if *key* exists and has not expired."""
        return self.get(key, _SENTINEL) is not _SENTINEL

    def __len__(self) -> int:
        """Return number of stored entries (including possibly expired ones)."""
        return len(self._store)

    def size(self) -> int:
        """Return count of non-expired entries."""
        now = time.monotonic()
        return sum(1 for _, exp in self._store.values() if exp > now)

    def evict_expired(self) -> int:
        """Proactively remove expired entries; return count removed."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if exp <= now]
        for k in expired:
            del self._store[k]
        return len(expired)

    def keys(self) -> list[str]:
        """Return non-expired keys in insertion order."""
        now = time.monotonic()
        return [k for k, (_, exp) in self._store.items() if exp > now]


_SENTINEL = object()


def memoize_ttl(ttl: float = 60.0) -> Any:
    """Decorator factory that caches function results by positional args for *ttl* seconds.

    Args:
        ttl: Cache lifetime in seconds.

    Returns:
        Decorator that wraps the function with a TTL cache.
    """

    def decorator(func: Any) -> Any:
        cache: TTLCache = TTLCache(ttl=ttl)

        def wrapper(*args: Any) -> Any:
            key = str(args)
            result = cache.get(key, _SENTINEL)
            if result is _SENTINEL:
                result = func(*args)
                cache.set(key, result)
            return result

        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        wrapper.__cache__ = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator
