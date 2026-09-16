"""Thread-safe TTL/LRU in-process cache for Ticket-Oracle predictions."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL expiry.

    Args:
        maxsize: Maximum number of entries to hold in memory.
        ttl: Time-to-live in seconds for each cache entry.
    """

    def __init__(self, maxsize: int = 512, ttl: float = 300.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return the cached value for key, or None if missing or expired.

        Args:
            key: Cache lookup key.

        Returns:
            Cached value, or None on miss or expiry.
        """
        with self._lock:
            if key not in self._store:
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                logger.debug("cache_expired", extra={"key": key})
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Store a value under key with the configured TTL.

        Args:
            key: Cache storage key.
            value: Value to cache.
        """
        with self._lock:
            expires_at = time.monotonic() + self._ttl
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            while len(self._store) > self._maxsize:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("cache_evicted", extra={"key": evicted_key})

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache.

        Args:
            key: Key to remove.
        """
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Evict all entries from the cache."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries (including potentially-expired ones)."""
        with self._lock:
            return len(self._store)


_prediction_cache = TTLCache(maxsize=512, ttl=300.0)


def get_prediction_cache() -> TTLCache:
    """Return the singleton prediction TTL cache."""
    return _prediction_cache
