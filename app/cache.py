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
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with the configured TTL."""
        if len(self._store) >= self._max:
            # Evict the oldest entry
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, time.monotonic() + self._ttl)

    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Empty the cache."""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def evict_expired(self) -> int:
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._store)

    def keys(self) -> list[str]:
        """Return a snapshot of currently stored keys (may include expired entries)."""
        return list(self._store.keys())

    @property
    def ttl(self) -> float:
        """Return the configured TTL in seconds."""
        return self._ttl

    @property
    def max_size(self) -> int:
        """Return the configured maximum number of entries."""
        return self._max_size


prediction_cache = TTLCache(ttl_seconds=30, max_size=500)
