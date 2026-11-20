"""Thread-safe TTL/LRU prediction cache for Sport-Cast."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 300
_DEFAULT_MAX_SIZE = 512


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL expiry.

    Args:
        ttl: Time-to-live in seconds for each cached entry.
        max_size: Maximum number of entries to retain.
    """

    def __init__(self, ttl: int = _DEFAULT_TTL, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, payload: dict) -> str:
        """Compute a stable SHA-256 cache key from a dict payload.

        Args:
            payload: Dictionary to hash.

        Returns:
            Hex-encoded SHA-256 key string.
        """
        serialised = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()

    def get(self, payload: dict) -> object | None:
        """Retrieve a cached result for the given payload.

        Args:
            payload: Request payload dict used as cache key.

        Returns:
            Cached value or None if missing/expired.
        """
        key = self._make_key(payload)
        with self._lock:
            if key not in self._store:
                return None
            ts, value = self._store[key]
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, payload: dict, value: object) -> None:
        """Store a result in the cache.

        Args:
            payload: Request payload dict used as cache key.
            value: Result to cache.
        """
        key = self._make_key(payload)
        with self._lock:
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("Cache evicted key %s", evicted_key[:8])

    def invalidate(self, payload: dict) -> bool:
        """Remove a specific entry from the cache.

        Args:
            payload: Request payload dict whose cache entry to remove.

        Returns:
            True if an entry was removed, False if it was not present.
        """
        key = self._make_key(payload)
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Return the current number of cached entries."""
        with self._lock:
            return len(self._store)


_prediction_cache = TTLCache(ttl=_DEFAULT_TTL)


def get_prediction_cache() -> TTLCache:
    """Return the module-level prediction cache singleton.

    Returns:
        The shared TTLCache instance for prediction results.
    """
    return _prediction_cache
