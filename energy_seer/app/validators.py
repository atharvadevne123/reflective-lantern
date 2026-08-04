"""Input validation helpers for Energy-Seer API."""

from __future__ import annotations


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))


def validate_meter_id(meter_id: str) -> str:
    """Validate and normalise a meter ID."""
    meter_id = meter_id.strip()
    if not meter_id:
        raise ValueError("meter_id must not be empty")
    if len(meter_id) > 64:
        raise ValueError("meter_id must be <= 64 characters")
    return meter_id


def validate_feature_window(values: list[float], min_length: int = 5) -> list[float]:
    """Ensure a feature window has enough data points for statistical tests."""
    if len(values) < min_length:
        raise ValueError(
            f"Feature window must have at least {min_length} values, got {len(values)}"
        )
    return values


def validate_forecast_length(n: int, max_length: int = 8760) -> int:
    """Validate a forecast horizon length.

    Args:
        n: Number of forecast steps requested.
        max_length: Upper bound (default 8760 = one year of hourly data).

    Returns:
        Validated *n*.

    Raises:
        ValueError: If *n* < 1 or *n* > *max_length*.
    """
    if n < 1:
        raise ValueError(f"forecast length must be at least 1, got {n}")
    if n > max_length:
        raise ValueError(f"forecast length {n} exceeds maximum {max_length}")
    return n


def validate_tariff(tariff: float) -> float:
    """Validate an electricity tariff in currency per kWh.

    Args:
        tariff: Cost per kWh (must be non-negative and finite).

    Returns:
        Validated tariff value.

    Raises:
        ValueError: If *tariff* is negative or non-finite.
    """
    import math

    if not math.isfinite(tariff):
        raise ValueError(f"tariff must be finite, got {tariff}")
    if tariff < 0:
        raise ValueError(f"tariff must be non-negative, got {tariff}")
    return tariff


def validate_readings_list(readings: list[float], allow_negative: bool = False) -> list[float]:
    """Validate a list of energy readings.

    Args:
        readings: List of kWh readings.
        allow_negative: Whether negative values are permitted.

    Returns:
        Validated list.

    Raises:
        ValueError: If the list is empty, contains non-finite values, or
            contains negatives when *allow_negative* is False.
    """
    import math

    if not readings:
        raise ValueError("readings list must not be empty")
    for i, v in enumerate(readings):
        if not math.isfinite(v):
            raise ValueError(f"non-finite value at index {i}: {v}")
        if not allow_negative and v < 0:
            raise ValueError(f"negative reading at index {i}: {v}")
    return readings


__all__ = [
    "clamp",
    "validate_feature_window",
    "validate_forecast_length",
    "validate_meter_id",
    "validate_readings_list",
    "validate_tariff",
]
