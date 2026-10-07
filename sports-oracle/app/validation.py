"""Input validation utilities for Sports-Oracle."""

from __future__ import annotations

from typing import Any


def validate_probability(value: float, name: str) -> float:
    """Validate a probability value is in [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return float(value)


def validate_positive(value: float, name: str) -> float:
    """Validate a value is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


def validate_non_negative_int(value: int, name: str) -> int:
    """Validate an integer is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


def validate_match_features(features: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete set of match feature values."""
    validated: dict[str, Any] = {}

    for field in ("home_form", "away_form"):
        if field in features:
            validated[field] = validate_probability(features[field], field)

    for field in ("home_attack", "away_attack", "home_defense", "away_defense"):
        if field in features:
            validated[field] = validate_positive(features[field], field)

    for field in ("h2h_home_wins", "h2h_draws", "h2h_away_wins"):
        if field in features:
            validated[field] = validate_non_negative_int(features[field], field)

    for field in ("home_rest_days", "away_rest_days"):
        if field in features:
            val = int(features[field])
            if not 1 <= val <= 30:
                raise ValueError(f"{field} must be between 1 and 30, got {val}")
            validated[field] = val

    return {**features, **validated}
