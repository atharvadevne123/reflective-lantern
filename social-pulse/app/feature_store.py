"""In-memory feature store for caching recent post feature vectors."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any


logger = logging.getLogger(__name__)


class FeatureCache:
    """LRU feature cache keyed by content hash.

    Stores the last *max_size* feature dicts in memory, automatically
    evicting the oldest entries when capacity is exceeded.

    Args:
        max_size: Maximum number of entries to cache.
        ttl_seconds: How long an entry is valid before expiry.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[dict[str, Any], float]] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached feature dict for *key*, or None on miss.

        Expired entries are removed on access.

        Args:
            key: Cache key (typically a content hash).

        Returns:
            The cached feature dict, or None if absent or expired.
        """
        if key not in self._store:
            self._misses += 1
            return None

        value, ts = self._store[key]
        if time.monotonic() - ts > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            logger.debug("Cache expired: key=%s", key)
            return None

        # Move to end (most recently used)
        self._store.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store *value* under *key*, evicting the oldest entry if full.

        Args:
            key: Cache key (typically a content hash).
            value: Feature dict to cache.
        """
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.monotonic())

        while len(self._store) > self.max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("Cache evicted: key=%s", evicted_key)

    def invalidate(self, key: str) -> bool:
        """Remove *key* from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            True if the key was present and removed, False if absent.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Empty the cache and reset statistics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that found a valid cached value."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict[str, Any]:
        """Return cache statistics as a dict.

        Returns:
            Dict with size, hits, misses, and hit_rate.
        """
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }
