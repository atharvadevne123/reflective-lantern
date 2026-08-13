"""
In-process TTL cache with bounded size for Price-Prophet.

Uses :func:`time.monotonic` for expiry checks so it is unaffected by
system clock adjustments.  Not thread-safe; wrap with a lock if used
from multiple threads.
"""

from __future__ import annotations

import time
from typing import Any


class TTLCache:
    """Time-to-live cache with optional maximum size.

    When the cache exceeds *max_size* entries the oldest entry
    (by insertion timestamp) is evicted before the new one is stored.

    Parameters
    ----------
    ttl_seconds:
        Number of seconds an entry remains valid after being stored
        (default 300).
    max_size:
        Maximum number of entries; 0 means unlimited (default 1000).
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        # Each entry: (value, stored_at_monotonic)
        self._store: dict[str, tuple[Any, float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, stored_at: float) -> bool:
        return (time.monotonic() - stored_at) > self.ttl_seconds

    def _evict_oldest(self) -> None:
        """Remove the entry with the earliest stored_at timestamp."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k][1])
        del self._store[oldest_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` if absent/expired.

        Parameters
        ----------
        key:
            Cache key to look up.

        Returns
        -------
        object or None
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        value, stored_at = entry
        if self._is_expired(stored_at):
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*.

        If the cache is at capacity the oldest entry is evicted first.

        Parameters
        ----------
        key:
            Cache key.
        value:
            Value to cache.
        """
        # Evict if over max_size (and max_size is bounded)
        if self.max_size > 0 and key not in self._store:
            while len(self._store) >= self.max_size:
                self._evict_oldest()

        self._store[key] = (value, time.monotonic())

    def delete(self, key: str) -> bool:
        """Remove the entry for *key*.

        Parameters
        ----------
        key:
            Key to remove.

        Returns
        -------
        bool
            ``True`` if *key* was present and removed, ``False`` otherwise.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    def size(self) -> int:
        """Return the number of entries currently in the cache.

        Returns
        -------
        int
        """
        return len(self._store)

    def __len__(self) -> int:
        return self.size()
