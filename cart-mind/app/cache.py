"""Thread-safe in-memory TTL cache for recommendation and similarity results.

Recommendation and similarity lookups are read-heavy and repeat often for the same
user or seed item within a browsing session, so a short-lived cache in front of the
ensemble removes redundant inference work without risking stale results outliving
their usefulness.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300.0
DEFAULT_MAX_ENTRIES = 1000


class TTLCache:
    """LRU cache with per-entry time-to-live expiry.

    Entries are evicted on two conditions: when their age exceeds ``ttl_seconds``,
    and when the cache exceeds ``max_entries`` (least-recently-used first). Both
    reads and writes take the lock, so the cache is safe to share across the
    worker threads FastAPI uses for sync endpoints.

    Args:
        ttl_seconds: Lifetime of an entry, in seconds, measured from insertion.
        max_entries: Hard cap on stored entries before LRU eviction kicks in.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """Initialise the TTL cache with the given time-to-live and capacity limits."""
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None`` if absent or expired.

        Args:
            key: Cache key to look up.

        Returns:
            The cached value, or ``None`` on a miss or expiry.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            inserted_at, value = entry
            if time.monotonic() - inserted_at > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Insert or replace a cache entry, evicting the LRU entry if full.

        Args:
            key: Cache key to write.
            value: Value to store against the key.
        """
        with self._lock:
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                evicted, _ = self._store.popitem(last=False)
                logger.debug("Cache eviction (size cap): %s", evicted)

    def delete(self, key: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            True if the key existed and was removed, False if it was absent.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug("Cache entry deleted: %s", key)
                return True
            return False

    def size(self) -> int:
        """Return the current number of entries in the cache."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Drop every entry and reset hit/miss counters."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache size and hit-rate statistics.

        Returns:
            Mapping with entry count, hit count, miss count, and hit rate.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "ttl_seconds": self._ttl,
                "max_entries": self._max,
            }


RECOMMENDATION_CACHE = TTLCache()
SIMILARITY_CACHE = TTLCache()


def cache_hit_rate(cache: TTLCache) -> float:
    """Return the current hit rate of *cache* as a fraction in [0, 1].

    Args:
        cache: A TTLCache instance.

    Returns:
        Hit rate; 0.0 when no requests have been made.
    """
    s = cache.stats()
    return s["hit_rate"]


def evict_expired(cache: TTLCache) -> int:
    """Scan *cache* and evict all entries whose TTL has elapsed.

    Args:
        cache: A TTLCache instance to scan.

    Returns:
        Number of entries evicted.
    """
    evicted = 0
    with cache._lock:
        now = time.monotonic()
        expired_keys = [k for k, (ts, _) in cache._store.items() if now - ts > cache._ttl]
        for k in expired_keys:
            del cache._store[k]
            evicted += 1
    return evicted
