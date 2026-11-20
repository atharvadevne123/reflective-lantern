"""Stage 9 — TTL + LRU cache for high-frequency retrievals.

Query traffic follows a steep power law: a small set of questions accounts
for most volume. Serving those from cache removes the entire retrieval +
rerank + scoring cost for repeat queries, cutting both latency and compute
spend. Keys are normalized so trivially different phrasings share entries.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from typing import Any

_NORMALIZE = re.compile(r"[^a-z0-9 ]+")


def normalize_query(query: str) -> str:
    """Canonical cache key text: lowercase, punctuation-free, single-spaced."""
    text = _NORMALIZE.sub(" ", query.lower())
    return " ".join(text.split())


def cache_key(query: str) -> str:
    """Stable hash of the normalized query."""
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()[:32]


class RetrievalCache:
    """LRU cache with per-entry TTL and hit/miss accounting."""

    def __init__(self, max_size: int = 2048, ttl_seconds: float = 300.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Any | None:
        """Return the cached value for a query, or None on miss/expiry."""
        key = cache_key(query)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        stored_at, value = entry
        if time.monotonic() - stored_at > self.ttl_seconds:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def put(self, query: str, value: Any) -> None:
        """Store a value, evicting the least-recently-used entry when full."""
        key = cache_key(query)
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def invalidate_all(self) -> None:
        """Drop every entry (called after any index mutation so answers
        never reflect superseded document versions)."""
        self._store.clear()

    def stats(self) -> dict[str, float | int]:
        """Hit/miss counters and current size for the /metrics endpoint."""
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    def __len__(self) -> int:
        return len(self._store)


def cache_fill_ratio(cache: RetrievalCache) -> float:
    """Return the fraction of cache capacity currently used.

    Args:
        cache: A QueryCache instance.

    Returns:
        Ratio in [0.0, 1.0]; 0.0 for an empty cache.
    """
    if cache.max_size == 0:
        return 0.0
    return round(len(cache) / cache.max_size, 4)


def top_cached_queries(cache: RetrievalCache, n: int = 5) -> list[str]:
    """Return the *n* most-recently-used cache keys.

    Args:
        cache: A QueryCache instance.
        n: Maximum number of keys to return.

    Returns:
        List of keys from most-recent to least-recent, up to *n* entries.
    """
    keys = list(cache._store.keys())
    return list(reversed(keys[-n:]))
