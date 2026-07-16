"""Input validation utilities for energy load data."""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

VALID_HOURS = frozenset(range(24))
VALID_MONTHS = frozenset(range(1, 13))
VALID_DOW = frozenset(range(7))
MIN_TEMP_C = -40.0
MAX_TEMP_C = 60.0
MIN_HUMIDITY = 0.0
MAX_HUMIDITY = 100.0
MIN_LOAD_MW = 0.0
MAX_LOAD_MW = 50_000.0


def validate_temporal_fields(hour: int, day_of_week: int, month: int) -> list[str]:
    """Return list of validation error messages for temporal fields."""
    errors = []
    if hour not in VALID_HOURS:
        errors.append(f"hour must be 0-23, got {hour}")
    if day_of_week not in VALID_DOW:
        errors.append(f"day_of_week must be 0-6, got {day_of_week}")
    if month not in VALID_MONTHS:
        errors.append(f"month must be 1-12, got {month}")
    return errors


def validate_weather_fields(temperature_c: float, humidity_pct: float) -> list[str]:
    """Return list of validation error messages for weather fields."""
    errors = []
    if not (MIN_TEMP_C <= temperature_c <= MAX_TEMP_C):
        errors.append(f"temperature_c must be {MIN_TEMP_C}..{MAX_TEMP_C}, got {temperature_c}")
    if not (MIN_HUMIDITY <= humidity_pct <= MAX_HUMIDITY):
        errors.append(f"humidity_pct must be 0-100, got {humidity_pct}")
    return errors


def validate_load_series(loads: list[float]) -> list[str]:
    """Return list of validation error messages for a load series."""
    import math

    errors = []
    if not loads:
        return errors
    out_of_range = [v for v in loads if not math.isfinite(v) or not (MIN_LOAD_MW <= v <= MAX_LOAD_MW)]
    if out_of_range:
        errors.append(f"{len(out_of_range)} load values out of range or non-finite [0, 50000 MW]")
    nans = [i for i, v in enumerate(loads) if math.isnan(v)]
    if nans:
        errors.append(f"NaN values at indices: {nans[:5]}")
    return errors


MAX_BUILDING_ID_LEN = 64
MAX_BATCH_SIZE = 100
MAX_FEATURE_VECTOR_DIM = 512
MIN_CONSUMPTION_KWH = 0.0
MAX_CONSUMPTION_KWH = 100_000.0


def validate_building_id(building_id: str) -> list[str]:
    """Return validation errors for a building_id string.

    Args:
        building_id: Unique building identifier to validate.

    Returns:
        List of error strings (empty when valid).
    """
    errors = []
    if not building_id:
        errors.append("building_id must not be empty")
    elif len(building_id) > MAX_BUILDING_ID_LEN:
        errors.append(f"building_id exceeds max length {MAX_BUILDING_ID_LEN}")
    elif not building_id.replace("-", "").replace("_", "").isalnum():
        errors.append("building_id must be alphanumeric with hyphens/underscores only")
    return errors


def validate_batch_size(n: int) -> list[str]:
    """Return validation errors when batch size *n* exceeds MAX_BATCH_SIZE.

    Args:
        n: Number of items in the batch.

    Returns:
        List of error strings (empty when valid).
    """
    if n > MAX_BATCH_SIZE:
        return [f"batch size {n} exceeds maximum {MAX_BATCH_SIZE}"]
    if n <= 0:
        return [f"batch size must be positive, got {n}"]
    return []


def validate_consumption_kwh(value: float) -> list[str]:
    """Return validation errors for a consumption_kwh reading.

    Args:
        value: Energy consumption value in kWh.

    Returns:
        List of error strings (empty when valid).
    """
    import math

    errors = []
    if not math.isfinite(value):
        errors.append(f"consumption_kwh must be finite, got {value}")
    elif not (MIN_CONSUMPTION_KWH <= value <= MAX_CONSUMPTION_KWH):
        errors.append(f"consumption_kwh must be {MIN_CONSUMPTION_KWH}..{MAX_CONSUMPTION_KWH}, got {value}")
    return errors


def validate_feature_vector(vector: list[float], expected_dim: int | None = None) -> list[str]:
    """Return validation errors for a numeric feature vector.

    Args:
        vector: List of float values representing a feature vector.
        expected_dim: When provided, check the vector has this exact length.

    Returns:
        List of error strings (empty when valid).
    """
    import math

    errors = []
    if not vector:
        errors.append("feature vector must not be empty")
        return errors
    if len(vector) > MAX_FEATURE_VECTOR_DIM:
        errors.append(f"feature vector dim {len(vector)} exceeds max {MAX_FEATURE_VECTOR_DIM}")
    if expected_dim is not None and len(vector) != expected_dim:
        errors.append(f"expected feature vector dim {expected_dim}, got {len(vector)}")
    non_finite = [i for i, v in enumerate(vector) if not math.isfinite(v)]
    if non_finite:
        errors.append(f"non-finite values at indices: {non_finite[:5]}")
    return errors


def batch_validate_readings(
    readings: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Validate a batch of energy readings, returning per-row error reports.

    Each reading dict should contain at minimum: 'hour', 'day_of_week', 'month',
    'temperature_c', 'humidity_pct', 'consumption_kwh'.

    Args:
        readings: List of reading dicts to validate.

    Returns:
        List of dicts with keys 'index', 'errors', 'valid'. Only rows with
        errors have non-empty 'errors' lists.
    """
    results = []
    for idx, row in enumerate(readings):
        errors: list[str] = []
        errors += validate_temporal_fields(
            int(row.get("hour", 0)),
            int(row.get("day_of_week", 0)),
            int(row.get("month", 1)),
        )
        errors += validate_weather_fields(
            float(row.get("temperature_c", 0.0)),
            float(row.get("humidity_pct", 0.0)),
        )
        errors += validate_consumption_kwh(float(row.get("consumption_kwh", 0.0)))
        results.append({"index": idx, "errors": errors, "valid": len(errors) == 0})
    return results


def is_weekend(day_of_week: int) -> bool:
    """Return True if day_of_week is Saturday (5) or Sunday (6)."""
    return day_of_week >= 5


def extract_temporal_from_datetime(dt: datetime) -> dict[str, object]:
    """Extract hour, day_of_week, month, is_weekend from a datetime."""
    return {
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "month": dt.month,
        "is_weekend": is_weekend(dt.weekday()),
    }
