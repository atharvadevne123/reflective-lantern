"""Simple in-memory TTL cache for prediction results."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe TTL cache for single-process use."""

    def __init__(self, ttl_seconds: int = 60, max_size: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max = max_size
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or None if absent or expired."""
        with self._lock:
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
        with self._lock:
            if len(self._store) >= self._max:
                oldest = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest]
                self.evictions += 1
            self._store[key] = (value, time.monotonic() + self._ttl)

    def invalidate(self, key: str) -> None:
        """Remove *key* from the cache (no-op if absent)."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Empty the cache and reset hit/miss/eviction counters."""
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def evict_expired(self) -> int:
        """Remove all expired entries and return the count removed."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]
        return len(expired)

    def __contains__(self, key: str) -> bool:
        """Return True if *key* is present and not expired."""
        return self.get(key) is not None

    @property
    def eviction_count(self) -> int:
        """Total number of LRU evictions since creation or last clear()."""
        return self.evictions

    @property
    def size(self) -> int:
        """Current number of entries (including potentially expired ones)."""
        with self._lock:
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
            "evictions": self.eviction_count,
        }


prediction_cache = TTLCache(ttl_seconds=30, max_size=500)

__all__ = [
    "TTLCache",
    "batch_get",
    "batch_set",
    "build_cache_key",
    "cache_hit_rate",
    "cache_key_from_dict",
    "cache_stats_summary",
    "evict_expired_keys",
    "prediction_cache",
    "warm_cache",
]


def build_cache_key(*parts: object) -> str:
    """Build a string cache key from multiple parts.

    Args:
        *parts: Objects whose string representations form the key.

    Returns:
        Colon-delimited string key.
    """
    return ":".join(str(p) for p in parts)


def evict_expired_keys(cache: TTLCache) -> int:
    """Force expiry sweep on a TTLCache and return the number of evicted entries.

    Args:
        cache: A TTLCache instance.

    Returns:
        Number of expired keys removed.
    """
    before = len(cache._store)
    cache.evict_expired()
    after = len(cache._store)
    return before - after


def cache_hit_rate(hits: int, misses: int) -> float:
    """Compute cache hit rate as a fraction in [0, 1].

    Args:
        hits: Number of cache hits.
        misses: Number of cache misses.

    Returns:
        Hit rate in [0, 1]. Returns 0.0 when hits + misses == 0.

    Raises:
        ValueError: If hits or misses are negative.
    """
    if hits < 0 or misses < 0:
        raise ValueError("hits and misses must be non-negative")
    total = hits + misses
    if total == 0:
        return 0.0
    return round(hits / total, 4)


def warm_cache(cache: TTLCache, items: dict) -> int:
    """Populate a TTLCache with a batch of items.

    Args:
        cache: Target TTLCache instance.
        items: Mapping of key -> value to pre-populate.

    Returns:
        Number of items inserted.
    """
    for k, v in items.items():
        cache.set(k, v)
    return len(items)


def cache_key_from_dict(params: dict[str, object], prefix: str = "") -> str:
    """Build a deterministic cache key from a parameter dictionary.

    Sorts keys alphabetically and joins them as ``key=value`` pairs separated
    by colons. A *prefix* is prepended (with a colon separator) when provided.

    Args:
        params: Dict of parameters to encode into the key.
        prefix: Optional namespace prefix for the key.

    Returns:
        A string suitable for use as a cache key.
    """
    parts = [f"{k}={params[k]}" for k in sorted(params)]
    key = ":".join(parts)
    return f"{prefix}:{key}" if prefix else key


def cache_stats_summary(cache: TTLCache) -> dict[str, object]:
    """Return a summary dict of hit/miss/eviction statistics for *cache*.

    Args:
        cache: The :class:`TTLCache` instance to inspect.

    Returns:
        Dict with 'hits', 'misses', 'evictions', 'hit_rate', and 'size'.
    """
    return {
        "hits": cache.hits,
        "misses": cache.misses,
        "evictions": cache.eviction_count,
        "hit_rate": round(cache.hit_rate, 4),
        "size": cache.size,
    }


def batch_get(cache: TTLCache, keys: list[str]) -> dict[str, object]:
    """Retrieve multiple keys from *cache* in one call.

    Args:
        cache: The :class:`TTLCache` instance to query.
        keys: List of cache key strings.

    Returns:
        Dict mapping each key to its cached value; missing/expired keys
        are absent from the result.
    """
    result: dict[str, object] = {}
    for key in keys:
        val = cache.get(key)
        if val is not None:
            result[key] = val
    return result


def batch_set(cache: TTLCache, items: dict[str, object]) -> None:
    """Store multiple key-value pairs in *cache* at once.

    Args:
        cache: The :class:`TTLCache` instance to populate.
        items: Dict of key → value pairs to cache.
    """
    for key, value in items.items():
        cache.set(key, value)
