"""Simple in-process LRU cache for repeated feature vectors."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_prediction_cache: dict[str, dict[str, Any]] = {}
CACHE_MAX_SIZE = 1024

_cache_hits = 0
_cache_misses = 0


def _make_key(features: list[float]) -> str:
    """Hash a feature vector to a cache key."""
    rounded = tuple(round(f, 6) for f in features)
    return hashlib.md5(str(rounded).encode()).hexdigest()


def cached_predict(features: list[float]) -> dict[str, Any] | None:
    """Return a cached prediction result if available."""
    global _cache_hits, _cache_misses
    key = _make_key(features)
    result = _prediction_cache.get(key)
    if result is None:
        _cache_misses += 1
    else:
        _cache_hits += 1
    return result


def cache_prediction(features: list[float], result: dict[str, Any]) -> None:
    """Store a prediction result in the cache, evicting oldest if full."""
    if len(_prediction_cache) >= CACHE_MAX_SIZE:
        oldest_key = next(iter(_prediction_cache))
        del _prediction_cache[oldest_key]
        logger.debug("Cache eviction: removed key %s", oldest_key)
    key = _make_key(features)
    _prediction_cache[key] = result


def cache_stats() -> dict[str, int]:
    """Return current cache statistics including hit/miss counters."""
    return {
        "size": len(_prediction_cache),
        "max_size": CACHE_MAX_SIZE,
        "hits": _cache_hits,
        "misses": _cache_misses,
    }


def clear_cache() -> None:
    """Clear the prediction cache and reset hit/miss counters."""
    global _cache_hits, _cache_misses
    _prediction_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Prediction cache cleared")
