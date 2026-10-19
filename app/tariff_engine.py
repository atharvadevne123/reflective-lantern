"""Time-of-use tariff engine for electricity bill simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RatePeriod(str, Enum):
    """Time-of-use rate period classification."""

    OFF_PEAK = "off_peak"
    SHOULDER = "shoulder"
    PEAK = "peak"
    SUPER_PEAK = "super_peak"


@dataclass
class TariffRate:
    """A single rate band within a time-of-use tariff schedule."""

    period: RatePeriod
    rate_per_kwh: float
    start_hour: int
    end_hour: int


@dataclass
class BillEstimate:
    """Electricity bill estimate for a consumption period."""

    total_cost: float
    energy_cost: float
    standing_charge: float
    kwh_by_period: dict[str, float]
    cost_by_period: dict[str, float]


DEFAULT_TOU_RATES: list[TariffRate] = [
    TariffRate(RatePeriod.OFF_PEAK, 0.08, 0, 7),
    TariffRate(RatePeriod.SHOULDER, 0.15, 7, 16),
    TariffRate(RatePeriod.PEAK, 0.30, 16, 21),
    TariffRate(RatePeriod.SHOULDER, 0.15, 21, 24),
]

DEFAULT_STANDING_CHARGE: float = 0.30
"""Daily standing charge in currency units."""


def classify_hour(hour: int, rates: list[TariffRate]) -> RatePeriod:
    """Return the rate period for a given hour-of-day.

    Args:
        hour: Hour of day in [0, 23].
        rates: Ordered list of :class:`TariffRate` bands.

    Returns:
        The :class:`RatePeriod` whose window contains ``hour``.

    Raises:
        ValueError: If hour is outside [0, 23] or no matching rate found.
    """
    if not (0 <= hour <= 23):
        raise ValueError(f"hour must be in [0, 23], got {hour}")
    for rate in rates:
        if rate.start_hour <= hour < rate.end_hour:
            return rate.period
    raise ValueError(f"No rate band covers hour {hour}")


def rate_for_hour(hour: int, rates: list[TariffRate]) -> float:
    """Return the per-kWh rate for a given hour-of-day.

    Args:
        hour: Hour of day in [0, 23].
        rates: Ordered list of :class:`TariffRate` bands.

    Returns:
        Rate in $/kWh.

    Raises:
        ValueError: If hour is outside [0, 23] or no matching rate found.
    """
    period = classify_hour(hour, rates)
    for rate in rates:
        if rate.period == period and rate.start_hour <= hour < rate.end_hour:
            return rate.rate_per_kwh
    raise ValueError(f"No rate found for hour {hour}")


def simulate_bill(
    hourly_kwh: list[float],
    rates: list[TariffRate] | None = None,
    standing_charge_daily: float = DEFAULT_STANDING_CHARGE,
    days: int = 1,
) -> BillEstimate:
    """Simulate an electricity bill from hourly consumption data.

    Args:
        hourly_kwh: List of 24 hourly kWh readings.
        rates: TOU rate schedule; defaults to :data:`DEFAULT_TOU_RATES`.
        standing_charge_daily: Daily fixed charge.
        days: Number of days this profile represents (scales standing charge).

    Returns:
        :class:`BillEstimate` with cost breakdown by rate period.

    Raises:
        ValueError: If hourly_kwh does not have exactly 24 entries.
    """
    if len(hourly_kwh) != 24:
        raise ValueError(f"hourly_kwh must have 24 entries, got {len(hourly_kwh)}")
    if rates is None:
        rates = DEFAULT_TOU_RATES
    kwh_by_period: dict[str, float] = {}
    cost_by_period: dict[str, float] = {}
    for hour, kwh in enumerate(hourly_kwh):
        period = classify_hour(hour, rates).value
        r = rate_for_hour(hour, rates)
        kwh_by_period[period] = kwh_by_period.get(period, 0.0) + kwh
        cost_by_period[period] = cost_by_period.get(period, 0.0) + kwh * r
    energy_cost = round(sum(cost_by_period.values()), 4)
    standing = round(standing_charge_daily * days, 4)
    return BillEstimate(
        total_cost=round(energy_cost + standing, 4),
        energy_cost=energy_cost,
        standing_charge=standing,
        kwh_by_period={k: round(v, 4) for k, v in kwh_by_period.items()},
        cost_by_period={k: round(v, 4) for k, v in cost_by_period.items()},
    )


def peak_shifting_savings(
    hourly_kwh: list[float],
    shift_kwh: float,
    from_hour: int,
    to_hour: int,
    rates: list[TariffRate] | None = None,
) -> float:
    """Estimate bill savings from shifting load from a peak to an off-peak hour.

    Args:
        hourly_kwh: Original 24-hour consumption profile in kWh.
        shift_kwh: kWh to move between hours.
        from_hour: Source hour (expected to be a high-rate period).
        to_hour: Destination hour (expected to be a low-rate period).
        rates: TOU schedule; defaults to :data:`DEFAULT_TOU_RATES`.

    Returns:
        Bill savings in currency units (positive = savings).

    Raises:
        ValueError: If hours are out of range or shift_kwh is negative.
    """
    if shift_kwh < 0:
        raise ValueError("shift_kwh must be non-negative")
    if not (0 <= from_hour <= 23) or not (0 <= to_hour <= 23):
        raise ValueError("Hours must be in [0, 23]")
    if rates is None:
        rates = DEFAULT_TOU_RATES
    from_rate = rate_for_hour(from_hour, rates)
    to_rate = rate_for_hour(to_hour, rates)
    return round((from_rate - to_rate) * shift_kwh, 4)
