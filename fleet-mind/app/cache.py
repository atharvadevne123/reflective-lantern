"""Simple in-process LRU cache for repeated fleet telemetry lookups."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_prediction_cache: dict[str, Any] = {}
_CACHE_MAX = 1000


def _make_cache_key(reading: dict[str, float]) -> str:
    """Produce a deterministic cache key from a telemetry reading dict.

    Args:
        reading: Dict mapping feature name to float value.

    Returns:
        A 16-character hex digest of the sorted JSON representation.
    """
    payload = json.dumps(reading, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]  # noqa: S324


def get_cached_prediction(reading: dict[str, float]) -> Any | None:
    """Return a cached prediction result, or None if not cached.

    Args:
        reading: Raw telemetry values (without vehicle_id).

    Returns:
        Previously computed prediction dict, or None.
    """
    key = _make_cache_key(reading)
    return _prediction_cache.get(key)


def set_cached_prediction(reading: dict[str, float], result: Any) -> None:
    """Store a prediction result under a derived cache key.

    Args:
        reading: Raw telemetry values used to compute the result.
        result: Prediction dict to cache.
    """
    if len(_prediction_cache) >= _CACHE_MAX:
        # Evict the oldest entry (FIFO approximation)
        oldest_key = next(iter(_prediction_cache))
        del _prediction_cache[oldest_key]
    key = _make_cache_key(reading)
    _prediction_cache[key] = result
    logger.debug("Cache set for key %s (size=%d)", key, len(_prediction_cache))


def clear_cache() -> None:
    """Flush all cached predictions (called on model retrain)."""
    _prediction_cache.clear()
    logger.info("Prediction cache cleared")


def cache_size() -> int:
    """Return the number of entries currently in the prediction cache."""
    return len(_prediction_cache)
