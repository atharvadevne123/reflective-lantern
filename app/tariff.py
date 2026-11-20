"""Time-of-use tariff modelling and electricity cost computation.

Provides tariff schedules (flat, time-of-use, tiered) and helpers to price
an hourly consumption series under each scheme, plus a comparison utility
that recommends the cheapest scheme for a given load profile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_FLAT_RATE: float = 0.15
"""Default flat electricity rate in currency per kWh."""

DEFAULT_PEAK_HOURS: tuple[int, ...] = (16, 17, 18, 19, 20)
"""Hours of the day (0-23) treated as on-peak by default."""

DEFAULT_PEAK_RATE: float = 0.32
DEFAULT_OFF_PEAK_RATE: float = 0.09


@dataclass
class TieredBand:
    """A single consumption band in a tiered tariff.

    Args:
        limit_kwh: Upper bound of this band in kWh. ``None`` means unbounded.
        rate: Price per kWh applied to consumption inside this band.
    """

    limit_kwh: float | None
    rate: float


@dataclass
class TariffComparison:
    """Result of pricing one load profile under several tariff schemes."""

    flat_cost: float
    time_of_use_cost: float
    tiered_cost: float
    cheapest_scheme: str
    saving_vs_flat: float
    hours_priced: int = field(default=0)


def flat_rate_cost(hourly_kwh: list[float], rate: float = DEFAULT_FLAT_RATE) -> float:
    """Price a consumption series under a single flat rate.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.
        rate: Price per kWh.

    Returns:
        Total cost rounded to 2 decimal places.

    Raises:
        ValueError: If *rate* is negative or any consumption value is negative.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    total_kwh = _validated_total(hourly_kwh)
    cost = round(total_kwh * rate, 2)
    logger.debug("Flat-rate cost: %.3f kWh x %.4f = %.2f", total_kwh, rate, cost)
    return cost


def time_of_use_cost(
    hourly_kwh: list[float],
    start_hour: int = 0,
    peak_hours: tuple[int, ...] = DEFAULT_PEAK_HOURS,
    peak_rate: float = DEFAULT_PEAK_RATE,
    off_peak_rate: float = DEFAULT_OFF_PEAK_RATE,
) -> float:
    """Price a consumption series under a time-of-use tariff.

    Each entry in *hourly_kwh* is assigned a clock hour starting at
    *start_hour* and wrapping at 24. Entries falling on a peak hour are
    charged at *peak_rate*, all others at *off_peak_rate*.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.
        start_hour: Clock hour (0-23) of the first entry.
        peak_hours: Clock hours charged at the peak rate.
        peak_rate: Price per kWh during peak hours.
        off_peak_rate: Price per kWh outside peak hours.

    Returns:
        Total cost rounded to 2 decimal places.

    Raises:
        ValueError: If *start_hour* is outside 0-23, either rate is negative,
            or any consumption value is negative.
    """
    if not 0 <= start_hour <= 23:
        raise ValueError(f"start_hour must be in 0-23, got {start_hour}")
    if peak_rate < 0 or off_peak_rate < 0:
        raise ValueError(f"rates must be non-negative, got peak={peak_rate} off_peak={off_peak_rate}")
    _validated_total(hourly_kwh)

    peak_set = set(peak_hours)
    cost = 0.0
    for offset, kwh in enumerate(hourly_kwh):
        hour = (start_hour + offset) % 24
        cost += kwh * (peak_rate if hour in peak_set else off_peak_rate)
    return round(cost, 2)


def tiered_cost(hourly_kwh: list[float], bands: list[TieredBand] | None = None) -> float:
    """Price total consumption under a tiered (block) tariff.

    Consumption is accumulated across the whole series and then charged
    band by band: the portion falling inside each band is priced at that
    band's rate.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.
        bands: Ordered bands from lowest to highest. The final band should
            carry ``limit_kwh=None`` to absorb any remaining consumption.
            Defaults to a three-band residential schedule.

    Returns:
        Total cost rounded to 2 decimal places.

    Raises:
        ValueError: If *bands* is empty or any consumption value is negative.
    """
    if bands is None:
        bands = [
            TieredBand(limit_kwh=500.0, rate=0.11),
            TieredBand(limit_kwh=1500.0, rate=0.18),
            TieredBand(limit_kwh=None, rate=0.27),
        ]
    if not bands:
        raise ValueError("bands must not be empty")

    remaining = _validated_total(hourly_kwh)
    cost = 0.0
    consumed = 0.0
    for band in bands:
        if remaining <= 0:
            break
        if band.limit_kwh is None:
            band_capacity = remaining
        else:
            band_capacity = max(0.0, band.limit_kwh - consumed)
        charged = min(remaining, band_capacity)
        cost += charged * band.rate
        consumed += charged
        remaining -= charged
    if remaining > 0:
        logger.warning("Tiered tariff bands did not cover %.3f kWh; add an unbounded band", remaining)
    return round(cost, 2)


def compare_tariffs(
    hourly_kwh: list[float],
    start_hour: int = 0,
    flat_rate: float = DEFAULT_FLAT_RATE,
) -> TariffComparison:
    """Price a load profile under all three schemes and pick the cheapest.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.
        start_hour: Clock hour (0-23) of the first entry, for time-of-use.
        flat_rate: Price per kWh for the flat scheme, used as the baseline.

    Returns:
        A :class:`TariffComparison` naming the cheapest scheme and the saving
        it delivers against the flat baseline.
    """
    flat = flat_rate_cost(hourly_kwh, rate=flat_rate)
    tou = time_of_use_cost(hourly_kwh, start_hour=start_hour)
    tiered = tiered_cost(hourly_kwh)

    options = {"flat": flat, "time_of_use": tou, "tiered": tiered}
    cheapest = min(options, key=lambda name: options[name])
    saving = round(flat - options[cheapest], 2)
    logger.info("Cheapest tariff: %s at %.2f (saves %.2f vs flat)", cheapest, options[cheapest], saving)
    return TariffComparison(
        flat_cost=flat,
        time_of_use_cost=tou,
        tiered_cost=tiered,
        cheapest_scheme=cheapest,
        saving_vs_flat=saving,
        hours_priced=len(hourly_kwh),
    )


def peak_shift_saving(
    hourly_kwh: list[float],
    shiftable_fraction: float,
    start_hour: int = 0,
    peak_hours: tuple[int, ...] = DEFAULT_PEAK_HOURS,
    peak_rate: float = DEFAULT_PEAK_RATE,
    off_peak_rate: float = DEFAULT_OFF_PEAK_RATE,
) -> float:
    """Estimate the saving from shifting peak load into off-peak hours.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.
        shiftable_fraction: Fraction (0-1) of peak-hour load that can move.
        start_hour: Clock hour (0-23) of the first entry.
        peak_hours: Clock hours charged at the peak rate.
        peak_rate: Price per kWh during peak hours.
        off_peak_rate: Price per kWh outside peak hours.

    Returns:
        Estimated saving rounded to 2 decimal places. Zero when the off-peak
        rate is not cheaper than the peak rate.

    Raises:
        ValueError: If *shiftable_fraction* is outside 0-1.
    """
    if not 0.0 <= shiftable_fraction <= 1.0:
        raise ValueError(f"shiftable_fraction must be in 0-1, got {shiftable_fraction}")
    if off_peak_rate >= peak_rate:
        return 0.0

    peak_set = set(peak_hours)
    peak_kwh = sum(kwh for offset, kwh in enumerate(hourly_kwh) if (start_hour + offset) % 24 in peak_set)
    shifted = peak_kwh * shiftable_fraction
    return round(shifted * (peak_rate - off_peak_rate), 2)


def _validated_total(hourly_kwh: list[float]) -> float:
    """Return the summed consumption, rejecting negative readings.

    Args:
        hourly_kwh: Consumption values in kWh.

    Returns:
        Sum of all values; ``0.0`` for an empty series.

    Raises:
        ValueError: If any value is negative.
    """
    for kwh in hourly_kwh:
        if kwh < 0:
            raise ValueError(f"consumption values must be non-negative, got {kwh}")
    return float(sum(hourly_kwh))


__all__ = [
    "DEFAULT_FLAT_RATE",
    "DEFAULT_OFF_PEAK_RATE",
    "DEFAULT_PEAK_HOURS",
    "DEFAULT_PEAK_RATE",
    "TariffComparison",
    "TieredBand",
    "compare_tariffs",
    "flat_rate_cost",
    "peak_shift_saving",
    "tiered_cost",
    "time_of_use_cost",
]
