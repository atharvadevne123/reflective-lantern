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
    errors = []
    if not loads:
        return errors
    out_of_range = [v for v in loads if not (MIN_LOAD_MW <= v <= MAX_LOAD_MW)]
    if out_of_range:
        errors.append(f"{len(out_of_range)} load values out of range [0, 50000 MW]")
    nans = [i for i, v in enumerate(loads) if v != v]
    if nans:
        errors.append(f"NaN values at indices: {nans[:5]}")
    return errors


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
