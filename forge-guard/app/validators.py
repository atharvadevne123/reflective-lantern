"""Domain-specific input validators for manufacturing sensor readings.

Validates individual sensor values against physical plausibility ranges
and cross-field consistency rules before forwarding to the ML model.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (-50.0, 300.0),
    "pressure": (0.0, 200.0),
    "vibration": (0.0, 50.0),
    "cycle_time": (0.1, 600.0),
    "tool_wear": (0.0, 500.0),
    "power_consumption": (0.0, 500.0),
    "humidity": (0.0, 100.0),
}


def validate_sensor_reading(data: dict[str, float]) -> list[str]:
    """Return a list of validation error messages for *data*.

    An empty list means the reading is valid. Checks:
    - All required fields present
    - All values finite (not NaN / Inf)
    - All values within physical plausibility ranges
    - Cross-field rules (e.g. power > 0 when cycle active)
    """
    errors: list[str] = []

    for field, (lo, hi) in SENSOR_RANGES.items():
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue
        val = data[field]
        if not math.isfinite(val):
            errors.append(f"{field} must be finite (got {val})")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{field} out of range [{lo}, {hi}] (got {val})")

    # Cross-field: high vibration with very low cycle time is physically improbable
    if not errors and data.get("vibration", 0) > 40 and data.get("cycle_time", 600) < 1.0:
        errors.append("Implausible combination: vibration > 40 mm/s with cycle_time < 1 s")

    return errors


def is_valid_sensor_reading(data: dict[str, float]) -> bool:
    """Return True if *data* passes all validation rules."""
    return len(validate_sensor_reading(data)) == 0


def sanitize_sensor_reading(data: dict[str, Any]) -> dict[str, float]:
    """Coerce all sensor values to float and fill missing fields with column medians.

    This is a best-effort sanitizer for batch jobs; prefer strict validation
    for real-time endpoints.
    """
    medians: dict[str, float] = {
        "temperature": 75.0,
        "pressure": 50.0,
        "vibration": 3.0,
        "cycle_time": 30.0,
        "tool_wear": 20.0,
        "power_consumption": 100.0,
        "humidity": 50.0,
    }
    result: dict[str, float] = {}
    for field in SENSOR_RANGES:
        raw = data.get(field)
        if raw is None:
            result[field] = medians[field]
            logger.debug("Filled missing field %s with median %.2f", field, medians[field])
        else:
            try:
                val = float(raw)
                lo, hi = SENSOR_RANGES[field]
                result[field] = max(lo, min(hi, val))
            except (TypeError, ValueError):
                result[field] = medians[field]
                logger.warning("Could not coerce %s=%r to float; using median.", field, raw)
    return result


def sensor_z_score(value: float, mean: float, std: float) -> float:
    """Return the Z-score of *value* relative to a population.

    Args:
        value: Observed sensor reading.
        mean: Population mean.
        std: Population standard deviation; must be positive.

    Returns:
        Signed Z-score.

    Raises:
        ValueError: If *std* is not positive.
    """
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    return (value - mean) / std


def clamp_to_range(field: str, value: float) -> float:
    """Clamp *value* to the valid sensor range for *field*.

    Args:
        field: Sensor field name (must be in SENSOR_RANGES).
        value: Raw reading to clamp.

    Returns:
        Value clamped to [lo, hi].

    Raises:
        KeyError: If *field* is not a recognised sensor.
    """
    if field not in SENSOR_RANGES:
        raise KeyError(f"Unknown sensor field: {field!r}")
    lo, hi = SENSOR_RANGES[field]
    return max(lo, min(hi, value))


def sensor_drift_detected(
    baseline: list[float],
    current: list[float],
    threshold: float = 0.15,
) -> bool:
    """Return True if the mean of *current* has drifted beyond *threshold* from *baseline*.

    Drift is defined as abs(mean(current) - mean(baseline)) / (mean(baseline) + 1e-9).

    Args:
        baseline: Historical sensor readings.
        current: Recent sensor readings.
        threshold: Relative drift threshold in (0, 1].

    Returns:
        True when relative drift exceeds the threshold.

    Raises:
        ValueError: If either list is empty.
    """
    if not baseline or not current:
        raise ValueError("baseline and current must be non-empty")
    mean_base = sum(baseline) / len(baseline)
    mean_curr = sum(current) / len(current)
    relative_drift = abs(mean_curr - mean_base) / (abs(mean_base) + 1e-9)
    return relative_drift > threshold
