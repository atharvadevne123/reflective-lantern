"""Seasonal adjustment factors for fleet maintenance planning."""

from __future__ import annotations

from datetime import date

SEASON_MONTHS: dict[str, list[int]] = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}

DEFAULT_SEASON_FACTORS: dict[str, float] = {
    "winter": 1.25,
    "spring": 1.00,
    "summer": 1.10,
    "autumn": 1.05,
}


def season_for_date(d: date) -> str:
    """Return season name for a given date (Northern Hemisphere)."""
    for season, months in SEASON_MONTHS.items():
        if d.month in months:
            return season
    return "spring"


def adjustment_factor(
    d: date,
    factors: dict[str, float] | None = None,
) -> float:
    """Return the seasonal adjustment multiplier for a date."""
    factors = factors or DEFAULT_SEASON_FACTORS
    season = season_for_date(d)
    return factors.get(season, 1.0)


def adjust_forecast(
    base_value: float,
    forecast_date: date,
    factors: dict[str, float] | None = None,
) -> float:
    """Return base_value scaled by the seasonal factor for forecast_date."""
    return base_value * adjustment_factor(forecast_date, factors)


def seasonal_maintenance_schedule(
    base_interval_days: int,
    year: int,
    factors: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Return recommended maintenance intervals adjusted for each season."""
    result = []
    for season in SEASON_MONTHS:
        factor = (factors or DEFAULT_SEASON_FACTORS).get(season, 1.0)
        adjusted_interval = max(1, round(base_interval_days / factor))
        result.append(
            {
                "season": season,
                "adjustment_factor": factor,
                "recommended_interval_days": adjusted_interval,
            }
        )
    return result
