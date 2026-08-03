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
        if errors:
            logger.debug("Validation failed for reading %d: %s", idx, errors)
        results.append({"index": idx, "errors": errors, "valid": len(errors) == 0})
    n_invalid = sum(1 for r in results if not r["valid"])
    if n_invalid:
        logger.warning("batch_validate_readings: %d/%d readings invalid", n_invalid, len(readings))
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


MAX_FORECAST_HORIZON = 8760  # one year of hourly data
MIN_FORECAST_HORIZON = 1


def validate_forecast_horizon(horizon: int, max_horizon: int = MAX_FORECAST_HORIZON) -> list[str]:
    """Return validation errors for a forecast horizon value.

    Args:
        horizon: Number of future steps to forecast.
        max_horizon: Maximum allowed horizon (default 8760 = 1 year of hours).

    Returns:
        List of error strings (empty when valid).
    """
    errors = []
    if horizon < MIN_FORECAST_HORIZON:
        errors.append(f"horizon must be at least {MIN_FORECAST_HORIZON}, got {horizon}")
    if horizon > max_horizon:
        errors.append(f"horizon {horizon} exceeds maximum {max_horizon}")
    return errors


def is_valid_temporal_input(hour: int, day_of_week: int, month: int) -> bool:
    """Return True if all temporal fields pass validation with no errors.

    A convenience wrapper around :func:`validate_temporal_fields` for callers
    that need a simple boolean result rather than a list of error messages.

    Args:
        hour: Hour of day (0-23).
        day_of_week: Day of week (0=Monday … 6=Sunday).
        month: Month of year (1-12).

    Returns:
        True when all fields are valid, False otherwise.
    """
    return not validate_temporal_fields(hour, day_of_week, month)


def clamp_consumption(value: float) -> float:
    """Clamp *value* to the valid consumption range [MIN_CONSUMPTION_KWH, MAX_CONSUMPTION_KWH].

    Args:
        value: Raw consumption reading that may be out of range.

    Returns:
        Clamped float within [0.0, 100000.0].
    """
    return max(MIN_CONSUMPTION_KWH, min(value, MAX_CONSUMPTION_KWH))


def validate_reading_dict(reading: dict[str, object]) -> dict[str, object]:
    """Run all validators against a single energy reading dict.

    Combines temporal, weather, and consumption validation into one call.

    Args:
        reading: Dict with energy-reading fields (hour, day_of_week, month,
            temperature_c, humidity_pct, consumption_kwh).

    Returns:
        Dict with ``valid`` (bool), ``errors`` (list[str]), and ``warnings`` (list[str]).
    """
    errors: list[str] = []

    hour = reading.get("hour", 0)
    dow = reading.get("day_of_week", 0)
    month = reading.get("month", 1)
    errors.extend(validate_temporal_fields(int(hour), int(dow), int(month)))

    temp = reading.get("temperature_c")
    hum = reading.get("humidity_pct")
    if temp is not None and hum is not None:
        errors.extend(validate_weather_fields(float(temp), float(hum)))

    kwh = reading.get("consumption_kwh")
    if kwh is not None:
        errors.extend(validate_consumption_kwh(float(kwh)))
    else:
        errors.append("consumption_kwh is required")

    building_id = reading.get("building_id")
    if building_id is not None:
        errors.extend(validate_building_id(str(building_id)))

    return {"valid": len(errors) == 0, "errors": errors, "warnings": []}


MAX_PRICE_USD = 1_000_000_000.0


def validate_price(price: float) -> list[str]:
    """Return validation errors for a price value in USD.

    Args:
        price: Dollar amount to validate.

    Returns:
        List of error strings (empty when valid).
    """
    import math

    errors: list[str] = []
    if not math.isfinite(price):
        errors.append(f"price must be finite, got {price}")
    elif price < 0:
        errors.append(f"price must be non-negative, got {price}")
    elif price > MAX_PRICE_USD:
        errors.append(f"price {price} exceeds maximum {MAX_PRICE_USD}")
    return errors


def validate_coordinate(lat: float, lon: float) -> list[str]:
    """Return validation errors for a geographic coordinate pair.

    Args:
        lat: Latitude in decimal degrees (-90 to 90).
        lon: Longitude in decimal degrees (-180 to 180).

    Returns:
        List of error strings (empty when valid).
    """
    errors: list[str] = []
    if not (-90.0 <= lat <= 90.0):
        errors.append(f"latitude must be in [-90, 90], got {lat}")
    if not (-180.0 <= lon <= 180.0):
        errors.append(f"longitude must be in [-180, 180], got {lon}")
    return errors


__all__ = [
    "batch_validate_readings",
    "clamp_consumption",
    "extract_temporal_from_datetime",
    "is_valid_temporal_input",
    "is_weekend",
    "validate_batch_size",
    "validate_building_id",
    "validate_consumption_kwh",
    "validate_coordinate",
    "validate_feature_vector",
    "validate_forecast_horizon",
    "validate_load_series",
    "validate_price",
    "validate_reading_dict",
    "validate_temporal_fields",
    "validate_weather_fields",
]
