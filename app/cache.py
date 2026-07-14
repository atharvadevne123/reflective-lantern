"""Simple in-memory TTL cache for prediction results."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-unsafe but lightweight TTL cache for single-process use."""

    def __init__(self, ttl_seconds: int = 60, max_size: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max = max_size
        self._store: dict[str, tuple[Any, float]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or None if absent or expired."""
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* with the configured TTL.

        Evicts the entry with the earliest expiry when the cache is full.
        """
        if len(self._store) >= self._max:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, time.monotonic() + self._ttl)

    def invalidate(self, key: str) -> None:
        """Remove *key* from the cache (no-op if absent)."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Empty the cache and reset hit/miss counters."""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def evict_expired(self) -> int:
        """Remove all expired entries and return the count removed."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        return len(expired)

    @property
    def size(self) -> int:
        """Current number of entries (including potentially expired ones)."""
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Fraction of get() calls that returned a value (0.0 if none yet)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_or_set(self, key: str, default: Any) -> Any:
        """Return cached value; if absent or expired, store *default* and return it."""
        value = self.get(key)
        if value is None:
            self.set(key, default)
            return default
        return value

    def stats(self) -> dict[str, int | float]:
        """Return a dict snapshot of cache performance counters."""
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }


prediction_cache = TTLCache(ttl_seconds=30, max_size=500)

__all__ = ["TTLCache", "prediction_cache"]
