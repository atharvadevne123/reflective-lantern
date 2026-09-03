"""Building load-profile characterisation.

Summarises an hourly consumption series into the shape metrics operators
use to describe a building's demand: base load, load factor, peak-to-average
ratio, ramp rate, and a coarse profile classification.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BASE_LOAD_PERCENTILE: float = 0.10
"""Quantile of the sorted series treated as the always-on base load."""

FLAT_LOAD_FACTOR_THRESHOLD: float = 0.80
PEAKY_LOAD_FACTOR_THRESHOLD: float = 0.45


@dataclass
class LoadProfile:
    """Shape metrics for one building's hourly demand series."""

    base_load_kwh: float
    peak_kwh: float
    mean_kwh: float
    load_factor: float
    peak_to_average: float
    max_ramp_kwh: float
    profile_class: str


def base_load(hourly_kwh: list[float], percentile: float = BASE_LOAD_PERCENTILE) -> float:
    """Estimate the always-on base load from the low tail of the series.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.
        percentile: Quantile (0-1) of the sorted series to read. The default
            of 0.10 ignores the very lowest readings as noise.

    Returns:
        Base load in kWh rounded to 4 decimal places.

    Raises:
        ValueError: If *hourly_kwh* is empty or *percentile* is outside 0-1.
    """
    if not hourly_kwh:
        raise ValueError("hourly_kwh must not be empty")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"percentile must be in 0-1, got {percentile}")
    ordered = sorted(hourly_kwh)
    index = min(int(percentile * len(ordered)), len(ordered) - 1)
    return round(float(ordered[index]), 4)


def load_factor(hourly_kwh: list[float]) -> float:
    """Return mean demand divided by peak demand.

    A load factor near 1.0 means flat, efficient utilisation; a low value
    means the building is sized for peaks it rarely reaches.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.

    Returns:
        Load factor in 0-1 rounded to 4 decimal places. Returns ``0.0`` when
        the peak is zero.

    Raises:
        ValueError: If *hourly_kwh* is empty.
    """
    if not hourly_kwh:
        raise ValueError("hourly_kwh must not be empty")
    peak = max(hourly_kwh)
    if peak <= 0:
        return 0.0
    return round(statistics.fmean(hourly_kwh) / peak, 4)


def peak_to_average_ratio(hourly_kwh: list[float]) -> float:
    """Return peak demand divided by mean demand.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.

    Returns:
        Peak-to-average ratio rounded to 4 decimal places. Returns ``0.0``
        when the mean is zero.

    Raises:
        ValueError: If *hourly_kwh* is empty.
    """
    if not hourly_kwh:
        raise ValueError("hourly_kwh must not be empty")
    mean = statistics.fmean(hourly_kwh)
    if mean <= 0:
        return 0.0
    return round(max(hourly_kwh) / mean, 4)


def max_ramp_rate(hourly_kwh: list[float]) -> float:
    """Return the largest absolute hour-over-hour change in demand.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.

    Returns:
        Largest absolute step in kWh rounded to 4 decimal places. Returns
        ``0.0`` for a series shorter than two entries.
    """
    if len(hourly_kwh) < 2:
        return 0.0
    steps = (abs(hourly_kwh[i + 1] - hourly_kwh[i]) for i in range(len(hourly_kwh) - 1))
    return round(max(steps), 4)


def classify_profile(load_factor_value: float) -> str:
    """Bucket a load factor into a coarse profile class.

    Args:
        load_factor_value: Load factor in 0-1.

    Returns:
        One of ``"flat"``, ``"moderate"``, or ``"peaky"``.
    """
    if load_factor_value >= FLAT_LOAD_FACTOR_THRESHOLD:
        return "flat"
    if load_factor_value >= PEAKY_LOAD_FACTOR_THRESHOLD:
        return "moderate"
    return "peaky"


def build_load_profile(hourly_kwh: list[float]) -> LoadProfile:
    """Compute the full set of shape metrics for a demand series.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.

    Returns:
        A populated :class:`LoadProfile`.

    Raises:
        ValueError: If *hourly_kwh* is empty.
    """
    if not hourly_kwh:
        raise ValueError("hourly_kwh must not be empty")
    lf = load_factor(hourly_kwh)
    profile = LoadProfile(
        base_load_kwh=base_load(hourly_kwh),
        peak_kwh=round(float(max(hourly_kwh)), 4),
        mean_kwh=round(statistics.fmean(hourly_kwh), 4),
        load_factor=lf,
        peak_to_average=peak_to_average_ratio(hourly_kwh),
        max_ramp_kwh=max_ramp_rate(hourly_kwh),
        profile_class=classify_profile(lf),
    )
    logger.info(
        "Load profile: class=%s load_factor=%.3f peak=%.2f base=%.2f",
        profile.profile_class,
        profile.load_factor,
        profile.peak_kwh,
        profile.base_load_kwh,
    )
    return profile


def demand_variability(hourly_kwh: list[float]) -> float:
    """Return coefficient of variation of the demand series.

    CV = std_dev / mean; measures volatility relative to average demand.
    Returns 0.0 for series with zero mean or fewer than 2 points.

    Args:
        hourly_kwh: Consumption values in kWh, one entry per hour.

    Returns:
        Coefficient of variation, rounded to 4 decimal places.
    """
    if len(hourly_kwh) < 2:
        return 0.0
    mean = statistics.fmean(hourly_kwh)
    if mean <= 0:
        return 0.0
    return round(statistics.stdev(hourly_kwh) / mean, 4)


def night_load_fraction(hourly_kwh: list[float], night_hours: tuple[int, int] = (22, 6)) -> float:
    """Return the fraction of total energy consumed during night hours.

    Args:
        hourly_kwh: 24-element list of hourly consumption values (index 0 = midnight).
        night_hours: Tuple (start_hour, end_hour) defining the night window.
            The window wraps midnight, e.g. (22, 6) covers 22:00–05:59.

    Returns:
        Fraction in [0, 1] rounded to 4 decimal places. Returns 0.0 if total
        consumption is zero or the series has fewer than 24 values.
    """
    if len(hourly_kwh) < 24:
        return 0.0
    total = sum(hourly_kwh[:24])
    if total <= 0:
        return 0.0
    start, end = night_hours
    if start > end:
        night = sum(hourly_kwh[start:24]) + sum(hourly_kwh[:end])
    else:
        night = sum(hourly_kwh[start:end])
    return round(night / total, 4)


__all__ = [
    "BASE_LOAD_PERCENTILE",
    "LoadProfile",
    "base_load",
    "build_load_profile",
    "classify_profile",
    "demand_variability",
    "load_factor",
    "max_ramp_rate",
    "night_load_fraction",
    "peak_to_average_ratio",
]
