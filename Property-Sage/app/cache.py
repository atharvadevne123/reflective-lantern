"""Simple in-memory prediction cache for Property-Sage."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class PredictionCache:
    """LRU-style TTL cache for prediction results."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        """Hash a payload dict into a cache key.

        Args:
            payload: Request payload dict.

        Returns:
            Hex-encoded SHA-256 hash string (first 16 chars).
        """
        serialised = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        """Look up a cache entry, returning None if absent or expired.

        Args:
            key: Cache key string.

        Returns:
            Cached value, or None if not present or expired.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value under key with TTL expiry.

        Args:
            key: Cache key string.
            value: Value to cache.
        """
        if len(self._store) >= self.max_size:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = (value, time.monotonic() + self.ttl_seconds)

    def invalidate(self, key: str) -> None:
        """Remove an entry from the cache.

        Args:
            key: Cache key to remove.
        """
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    def size(self) -> int:
        """Return the number of currently stored entries.

        Returns:
            Entry count.
        """
        return len(self._store)
