"""Thread-safe TTL/LRU in-memory cache."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """Thread-safe TTL/LRU cache with bounded size and automatic expiry."""

    def __init__(self, maxsize: int = 256, ttl_seconds: float = 300.0) -> None:
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing or expired."""
        with self._lock:
            if key not in self._store:
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Insert or refresh a cache entry, evicting LRU items when full."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.monotonic() + self._ttl)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a single key if present."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Evict all cached entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        """Return the number of live (potentially expired) entries."""
        return len(self._store)

    def stats(self) -> dict[str, int]:
        """Return current size and capacity metadata."""
        with self._lock:
            return {"size": len(self._store), "maxsize": self._maxsize}


    def peek(self, key: str) -> Any | None:
        """Return a cached value without updating the LRU order.

        Args:
            key: Cache key to inspect.

        Returns:
            Cached value if present and unexpired, otherwise None.
        """
        with self._lock:
            if key not in self._store:
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def keys(self) -> list[str]:
        """Return a snapshot of all live, unexpired cache keys."""
        now = time.monotonic()
        with self._lock:
            return [k for k, (_, exp) in list(self._store.items()) if exp > now]


def get_cache_key(*parts: object) -> str:
    """Build a deterministic cache key from an arbitrary sequence of values.

    Args:
        *parts: Values that will be str-joined with ``|`` as the separator.

    Returns:
        A single string suitable for use as a TTLCache key.
    """
    return "|".join(str(p) for p in parts)


_prediction_cache = TTLCache(maxsize=512, ttl_seconds=300)
