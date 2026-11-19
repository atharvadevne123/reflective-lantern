"""Predictive analytics helpers for remaining useful life and failure probability."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RULInput:
    current_value: float
    degradation_rate_per_km: float
    failure_threshold: float
    odometer_km: float = 0.0


def remaining_useful_life_km(inp: RULInput) -> float:
    """Return estimated km until value reaches failure threshold."""
    if inp.degradation_rate_per_km <= 0:
        return float("inf")
    headroom = inp.current_value - inp.failure_threshold
    if headroom <= 0:
        return 0.0
    return headroom / inp.degradation_rate_per_km


def weibull_failure_probability(
    age_km: float,
    scale: float,
    shape: float = 2.0,
) -> float:
    """Return Weibull CDF failure probability for a component at age_km."""
    if scale <= 0 or age_km < 0:
        return 0.0
    return 1.0 - math.exp(-((age_km / scale) ** shape))


def linear_degradation_forecast(
    current: float,
    rate_per_km: float,
    forecast_km: float,
) -> float:
    """Return linearly forecasted value after forecast_km additional km."""
    return current - rate_per_km * forecast_km


def time_to_threshold(
    current: float,
    threshold: float,
    rate_per_unit: float,
) -> float:
    """Return units until value declines to threshold at given rate."""
    if rate_per_unit <= 0:
        return float("inf")
    headroom = current - threshold
    if headroom <= 0:
        return 0.0
    return headroom / rate_per_unit


def maintenance_urgency_score(
    rul_km: float,
    max_rul_km: float = 50_000.0,
) -> float:
    """Return 0-1 urgency score (1 = needs immediate attention)."""
    if max_rul_km <= 0:
        return 1.0
    return max(0.0, 1.0 - min(1.0, rul_km / max_rul_km))
