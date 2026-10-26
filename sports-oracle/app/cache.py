"""Thread-safe TTL cache for Sports-Oracle predictions."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any


class TTLCache:
    """In-memory LRU+TTL cache for prediction results."""

    def __init__(self, maxsize: int = 512, ttl: float = 300.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def _make_key(self, features: dict) -> str:
        """Create a deterministic cache key from feature dict."""
        serialised = json.dumps(features, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def get(self, features: dict) -> Any | None:
        """Return cached value or None if missing/expired."""
        key = self._make_key(features)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._store[key]
                return None
            return value

    def set(self, features: dict, value: Any) -> None:
        """Store value with TTL expiry, evicting LRU if full."""
        key = self._make_key(features)
        with self._lock:
            if len(self._store) >= self.maxsize and key not in self._store:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[key] = (value, time.monotonic() + self.ttl)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


_prediction_cache = TTLCache(maxsize=512, ttl=300.0)


def get_prediction_cache() -> TTLCache:
    """Return the global prediction cache instance."""
    return _prediction_cache
