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
        raise ValueError(f"Feature window must have at least {min_length} values, got {len(values)}")
    return values
