"""Simple in-memory LRU cache for Risk-Sense predictions."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_prediction_cache: dict[str, tuple[float, float]] = {}  # key -> (default_prob, timestamp)
CACHE_TTL_SECONDS = 300.0
MAX_CACHE_SIZE = 1000


def _make_cache_key(features: dict[str, Any]) -> str:
    """Hash a feature dict to a stable cache key.

    Args:
        features: Credit application feature dictionary.

    Returns:
        Hex digest string to use as cache key.
    """
    canonical = json.dumps(features, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_cached_prediction(features: dict[str, Any]) -> float | None:
    """Return cached prediction if present and not expired.

    Args:
        features: Credit application feature dictionary.

    Returns:
        Cached default probability, or None if not found/expired.
    """
    key = _make_cache_key(features)
    entry = _prediction_cache.get(key)
    if entry is None:
        return None
    value, ts = entry
    if time.monotonic() - ts > CACHE_TTL_SECONDS:
        del _prediction_cache[key]
        logger.debug("Cache expired for key %s", key[:8])
        return None
    logger.debug("Cache hit for key %s", key[:8])
    return value


def cache_prediction(features: dict[str, Any], default_prob: float) -> None:
    """Store a prediction in the cache.

    Args:
        features: Credit application feature dictionary.
        default_prob: Predicted default probability to cache.
    """
    if len(_prediction_cache) >= MAX_CACHE_SIZE:
        # Evict oldest entry
        oldest_key = min(_prediction_cache, key=lambda k: _prediction_cache[k][1])
        del _prediction_cache[oldest_key]

    key = _make_cache_key(features)
    _prediction_cache[key] = (default_prob, time.monotonic())
    logger.debug("Cached prediction %.4f for key %s", default_prob, key[:8])


def clear_cache() -> None:
    """Clear all cached predictions."""
    _prediction_cache.clear()
    logger.info("Prediction cache cleared")


def cache_stats() -> dict[str, int]:
    """Return current cache statistics.

    Returns:
        Dict with 'size' and 'max_size' keys.
    """
    return {"size": len(_prediction_cache), "max_size": MAX_CACHE_SIZE}


@lru_cache(maxsize=128)
def cached_risk_thresholds(
    low_threshold: float = 0.2,
    medium_threshold: float = 0.5,
    high_threshold: float = 0.75,
) -> tuple[float, float, float]:
    """Return cached risk category thresholds.

    Args:
        low_threshold: Score below which risk is 'low'.
        medium_threshold: Score below which risk is 'medium'.
        high_threshold: Score below which risk is 'high'.

    Returns:
        Tuple of (low, medium, high) thresholds.
    """
    return (low_threshold, medium_threshold, high_threshold)
