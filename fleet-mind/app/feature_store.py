"""Simple in-process feature store for caching per-vehicle feature vectors."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_STORE: dict[str, dict[str, Any]] = {}
_TTL_SECONDS: float = 300.0  # 5 minutes


def store_features(
    vehicle_id: str,
    features: np.ndarray,
    ttl_seconds: float = _TTL_SECONDS,
) -> None:
    """Store a feature vector for a vehicle with a TTL.

    Args:
        vehicle_id: Unique vehicle identifier.
        features: Pre-computed feature array to cache.
        ttl_seconds: Time-to-live in seconds before the entry is considered stale.
    """
    _STORE[vehicle_id] = {
        "features": features,
        "expires_at": time.monotonic() + ttl_seconds,
    }
    logger.debug("Feature store: set %s (TTL=%.0fs)", vehicle_id, ttl_seconds)


def get_features(vehicle_id: str) -> np.ndarray | None:
    """Retrieve cached features for a vehicle, or None if expired or absent.

    Args:
        vehicle_id: Unique vehicle identifier.

    Returns:
        Cached feature array, or None if not found or expired.
    """
    entry = _STORE.get(vehicle_id)
    if entry is None:
        return None
    if time.monotonic() > entry["expires_at"]:
        del _STORE[vehicle_id]
        logger.debug("Feature store: expired %s", vehicle_id)
        return None
    return entry["features"]


def evict_expired() -> int:
    """Remove all expired entries from the store.

    Returns:
        Number of entries evicted.
    """
    now = time.monotonic()
    expired_keys = [k for k, v in _STORE.items() if now > v["expires_at"]]
    for key in expired_keys:
        del _STORE[key]
    if expired_keys:
        logger.debug("Feature store: evicted %d expired entries", len(expired_keys))
    return len(expired_keys)


def store_size() -> int:
    """Return the current number of entries in the feature store.

    Returns:
        Count of non-expired entries (expired entries may still be counted
        until evict_expired() is called).
    """
    return len(_STORE)
