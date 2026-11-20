"""Weather normalisation of energy consumption via degree days.

Heating and cooling degree days let you separate weather-driven demand from
genuine efficiency change, so a mild winter is not mistaken for a retrofit
paying off. Provides degree-day computation, a normalisation factor, and a
year-over-year comparison that reports weather-adjusted change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_BASE_TEMPERATURE_C: float = 18.0
"""Balance-point temperature below which heating is assumed to start."""


@dataclass
class NormalizedComparison:
    """Weather-adjusted comparison of two consumption periods."""

    baseline_kwh: float
    current_kwh: float
    raw_change_pct: float
    normalized_current_kwh: float
    normalized_change_pct: float
    weather_effect_pct: float


def heating_degree_days(
    daily_mean_temps_c: list[float],
    base_temperature_c: float = DEFAULT_BASE_TEMPERATURE_C,
) -> float:
    """Sum heating degree days across a series of daily mean temperatures.

    Each day contributes ``max(0, base - mean_temp)``.

    Args:
        daily_mean_temps_c: Daily mean outdoor temperature in Celsius.
        base_temperature_c: Balance-point temperature in Celsius.

    Returns:
        Total heating degree days rounded to 2 decimal places.
    """
    total = sum(max(0.0, base_temperature_c - temp) for temp in daily_mean_temps_c)
    return round(total, 2)


def cooling_degree_days(
    daily_mean_temps_c: list[float],
    base_temperature_c: float = DEFAULT_BASE_TEMPERATURE_C,
) -> float:
    """Sum cooling degree days across a series of daily mean temperatures.

    Each day contributes ``max(0, mean_temp - base)``.

    Args:
        daily_mean_temps_c: Daily mean outdoor temperature in Celsius.
        base_temperature_c: Balance-point temperature in Celsius.

    Returns:
        Total cooling degree days rounded to 2 decimal places.
    """
    total = sum(max(0.0, temp - base_temperature_c) for temp in daily_mean_temps_c)
    return round(total, 2)


def normalization_factor(baseline_degree_days: float, current_degree_days: float) -> float:
    """Return the factor that scales current demand to baseline weather.

    Args:
        baseline_degree_days: Degree days over the reference period.
        current_degree_days: Degree days over the period being assessed.

    Returns:
        ``baseline / current`` rounded to 4 decimal places, or ``1.0`` when
        *current_degree_days* is zero (no weather signal to correct for).

    Raises:
        ValueError: If either argument is negative.
    """
    if baseline_degree_days < 0 or current_degree_days < 0:
        raise ValueError(
            f"degree days must be non-negative, got baseline={baseline_degree_days} current={current_degree_days}"
        )
    if current_degree_days == 0:
        logger.debug("Current period has zero degree days; normalisation factor defaults to 1.0")
        return 1.0
    return round(baseline_degree_days / current_degree_days, 4)


def normalize_consumption(
    consumption_kwh: float,
    baseline_degree_days: float,
    current_degree_days: float,
) -> float:
    """Scale measured consumption to what it would be under baseline weather.

    Args:
        consumption_kwh: Measured consumption for the current period.
        baseline_degree_days: Degree days over the reference period.
        current_degree_days: Degree days over the current period.

    Returns:
        Weather-normalised consumption in kWh rounded to 3 decimal places.

    Raises:
        ValueError: If *consumption_kwh* is negative or either degree-day
            total is negative.
    """
    if consumption_kwh < 0:
        raise ValueError(f"consumption_kwh must be non-negative, got {consumption_kwh}")
    factor = normalization_factor(baseline_degree_days, current_degree_days)
    return round(consumption_kwh * factor, 3)


def compare_periods(
    baseline_kwh: float,
    current_kwh: float,
    baseline_degree_days: float,
    current_degree_days: float,
) -> NormalizedComparison:
    """Compare two periods, splitting raw change into weather and efficiency.

    ``weather_effect_pct`` is the share of the raw change attributable to
    differing weather; the remainder is the weather-adjusted change that
    reflects genuine operational or efficiency movement.

    Args:
        baseline_kwh: Consumption over the reference period.
        current_kwh: Consumption over the period being assessed.
        baseline_degree_days: Degree days over the reference period.
        current_degree_days: Degree days over the current period.

    Returns:
        A populated :class:`NormalizedComparison`.

    Raises:
        ValueError: If *baseline_kwh* is not positive, *current_kwh* is
            negative, or either degree-day total is negative.
    """
    if baseline_kwh <= 0:
        raise ValueError(f"baseline_kwh must be positive, got {baseline_kwh}")
    if current_kwh < 0:
        raise ValueError(f"current_kwh must be non-negative, got {current_kwh}")

    normalized = normalize_consumption(current_kwh, baseline_degree_days, current_degree_days)
    raw_change = round(100.0 * (current_kwh - baseline_kwh) / baseline_kwh, 2)
    normalized_change = round(100.0 * (normalized - baseline_kwh) / baseline_kwh, 2)
    weather_effect = round(raw_change - normalized_change, 2)

    logger.info(
        "Period comparison: raw %+.2f%%, weather-adjusted %+.2f%% (weather contributed %+.2f%%)",
        raw_change,
        normalized_change,
        weather_effect,
    )
    return NormalizedComparison(
        baseline_kwh=round(baseline_kwh, 3),
        current_kwh=round(current_kwh, 3),
        raw_change_pct=raw_change,
        normalized_current_kwh=normalized,
        normalized_change_pct=normalized_change,
        weather_effect_pct=weather_effect,
    )


__all__ = [
    "DEFAULT_BASE_TEMPERATURE_C",
    "NormalizedComparison",
    "compare_periods",
    "cooling_degree_days",
    "heating_degree_days",
    "normalization_factor",
    "normalize_consumption",
]
