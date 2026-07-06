"""In-memory TTL cache for expensive lookups in Realty-Edge.

Provides a simple dict-backed TTL cache to avoid repeated database queries
for neighbourhood stats and metrics that change infrequently.

Example
-------
cache = TTLCache(ttl_seconds=300)
cache.set("key", value)
result = cache.get("key")   # None after TTL expires
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-unsafe TTL cache backed by a plain dict.

    Suitable for single-process deployments. For multi-process or distributed
    deployments replace with Redis.

    Args:
        ttl_seconds: How long an entry remains valid after being set.
        max_size: Maximum number of entries; oldest entries are evicted first.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 512) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing or expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            logger.debug("Cache MISS (expired) key=%s", key)
            return None
        logger.debug("Cache HIT key=%s", key)
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with a TTL timestamp."""
        if len(self._store) >= self._max_size:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
            logger.debug("Cache evicted key=%s (max_size=%d)", oldest, self._max_size)
        self._store[key] = (value, time.monotonic() + self._ttl)
        logger.debug("Cache SET key=%s ttl=%.0fs", key, self._ttl)

    def invalidate(self, key: str) -> bool:
        """Remove a key from the cache. Returns True if the key existed."""
        existed = key in self._store
        self._store.pop(key, None)
        return existed

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


neighborhood_cache: TTLCache = TTLCache(ttl_seconds=300.0)
metrics_cache: TTLCache = TTLCache(ttl_seconds=60.0)
