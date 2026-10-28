"""Time-series analysis utilities for fleet telemetry trend detection."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Rolling window per vehicle per feature
_VEHICLE_WINDOWS: dict[str, dict[str, deque[float]]] = {}
_WINDOW_SIZE = 20


def _get_window(vehicle_id: str, feature: str) -> deque[float]:
    """Retrieve or create the rolling window for a vehicle feature pair.

    Args:
        vehicle_id: Unique vehicle identifier.
        feature: Telemetry feature name.

    Returns:
        deque holding the last _WINDOW_SIZE observations.
    """
    if vehicle_id not in _VEHICLE_WINDOWS:
        _VEHICLE_WINDOWS[vehicle_id] = {}
    if feature not in _VEHICLE_WINDOWS[vehicle_id]:
        _VEHICLE_WINDOWS[vehicle_id][feature] = deque(maxlen=_WINDOW_SIZE)
    return _VEHICLE_WINDOWS[vehicle_id][feature]


def update_window(vehicle_id: str, reading: dict[str, float]) -> None:
    """Push the latest reading into each feature's rolling window.

    Args:
        vehicle_id: The vehicle this reading belongs to.
        reading: Dict mapping feature name to float value.
    """
    for feature, value in reading.items():
        _get_window(vehicle_id, feature).append(value)


def rolling_stats(vehicle_id: str, feature: str) -> dict[str, float] | None:
    """Return rolling mean, std, and trend slope for a vehicle feature.

    Args:
        vehicle_id: The vehicle to compute stats for.
        feature: The telemetry feature of interest.

    Returns:
        Dict with 'mean', 'std', and 'slope', or None if fewer than 3 observations.
    """
    window = _get_window(vehicle_id, feature)
    if len(window) < 3:
        return None

    values = np.array(list(window))
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
    return {
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "slope": round(slope, 6),
    }


def detect_monotonic_degradation(
    vehicle_id: str,
    feature: str,
    min_slope: float = 0.05,
) -> bool:
    """Return True if a feature is trending monotonically (possible degradation).

    Args:
        vehicle_id: The vehicle to check.
        feature: The feature name to check for monotonic trend.
        min_slope: Minimum absolute slope to consider a trend significant.

    Returns:
        True if the trend slope exceeds min_slope in absolute value.
    """
    stats = rolling_stats(vehicle_id, feature)
    if stats is None:
        return False
    return abs(stats["slope"]) >= min_slope
