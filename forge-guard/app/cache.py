"""Thread-safe in-memory TTL cache for prediction results.

Reduces redundant model inference when the same sensor readings are
submitted repeatedly within the cache window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """A simple thread-safe dict-based cache with per-entry TTL expiry.

    Entries are evicted lazily on access and periodically via ``purge()``.

    Args:
        ttl_seconds: How long an entry remains valid after insertion.
        max_size: Maximum number of entries before oldest entries are dropped.
    """

    def __init__(self, ttl_seconds: int = 60, max_size: int = 1024) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def _make_key(self, data: dict[str, float]) -> str:
        """Produce a deterministic hash key from a sensor reading dict."""
        serialised = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()

    def get(self, data: dict[str, float]) -> Any | None:
        """Return cached result for *data* or None if missing/expired."""
        key = self._make_key(data)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._store[key]
                return None
            return value

    def set(self, data: dict[str, float], value: Any) -> None:
        """Store *value* under the hash of *data* with the configured TTL."""
        key = self._make_key(data)
        expiry = time.monotonic() + self.ttl_seconds
        with self._lock:
            if len(self._store) >= self.max_size:
                self._evict_oldest()
            self._store[key] = (value, expiry)

    def _evict_oldest(self) -> None:
        """Remove the entry with the earliest expiry time (called under lock)."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k][1])
        del self._store[oldest_key]

    def purge(self) -> int:
        """Remove all expired entries and return the count removed."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]
        if expired:
            logger.debug("Cache purged %d expired entries.", len(expired))
        return len(expired)

    def size(self) -> int:
        """Return the current number of entries (including expired ones)."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()


# Module-level singleton used by the prediction endpoints.
prediction_cache = TTLCache(ttl_seconds=int(30), max_size=2048)
