"""Peak demand analysis and demand-charge estimation for commercial buildings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_DEMAND_CHARGE_RATE: float = 12.0
"""Default demand charge in $/kW/month."""

DEFAULT_RATCHET_MONTHS: int = 12
"""Number of months over which ratchet clause looks back."""


@dataclass
class PeakDemandResult:
    """Result of a peak-demand analysis."""

    peak_kw: float
    average_kw: float
    load_factor: float
    demand_charge: float
    peak_index: int
    demand_charge_rate: float


def peak_kw(readings_kw: list[float]) -> float:
    """Return the maximum demand in kW from an interval readings list.

    Args:
        readings_kw: Time-series of demand readings in kW (non-negative).

    Returns:
        Maximum value, or 0.0 for an empty list.

    Raises:
        ValueError: If any reading is negative.
    """
    if not readings_kw:
        return 0.0
    for r in readings_kw:
        if r < 0:
            raise ValueError("Demand readings must be non-negative")
    return max(readings_kw)


def load_factor(readings_kw: list[float]) -> float:
    """Compute load factor as average_demand / peak_demand.

    A value close to 1.0 indicates flat, efficient load; near 0 means spiky.

    Args:
        readings_kw: Time-series of demand readings in kW.

    Returns:
        Load factor in [0, 1], or 0.0 when peak is zero.

    Raises:
        ValueError: If any reading is negative or the list is empty.
    """
    if not readings_kw:
        raise ValueError("readings_kw must not be empty")
    for r in readings_kw:
        if r < 0:
            raise ValueError("Demand readings must be non-negative")
    peak = max(readings_kw)
    if peak == 0.0:
        return 0.0
    avg = sum(readings_kw) / len(readings_kw)
    return round(avg / peak, 6)


def demand_charge(
    peak_demand_kw: float,
    rate_per_kw: float = DEFAULT_DEMAND_CHARGE_RATE,
) -> float:
    """Calculate monthly demand charge.

    Args:
        peak_demand_kw: Monthly peak demand in kW.
        rate_per_kw: Tariff rate in $/kW/month.

    Returns:
        Demand charge in dollars rounded to 2 decimal places.

    Raises:
        ValueError: If either argument is negative.
    """
    if peak_demand_kw < 0 or rate_per_kw < 0:
        raise ValueError("peak_demand_kw and rate_per_kw must be non-negative")
    return round(peak_demand_kw * rate_per_kw, 2)


def ratchet_demand(
    monthly_peaks_kw: list[float],
    ratchet_fraction: float = 0.85,
    n_months: int = DEFAULT_RATCHET_MONTHS,
) -> float:
    """Apply a ratchet clause to determine the billed demand.

    Many tariffs bill the higher of the current month's peak and a
    fraction of the highest peak in the preceding ``n_months``.

    Args:
        monthly_peaks_kw: List of monthly peak demand readings, most recent last.
        ratchet_fraction: Fraction applied to the historical maximum (default 85%).
        n_months: Look-back window in months (default 12).

    Returns:
        Billed demand in kW rounded to 4 decimal places.

    Raises:
        ValueError: If the list is empty or any value is negative.
    """
    if not monthly_peaks_kw:
        raise ValueError("monthly_peaks_kw must not be empty")
    for p in monthly_peaks_kw:
        if p < 0:
            raise ValueError("Peak demand values must be non-negative")
    current = monthly_peaks_kw[-1]
    history = monthly_peaks_kw[-(n_months + 1) : -1]
    if not history:
        return round(current, 4)
    ratchet = max(history) * ratchet_fraction
    return round(max(current, ratchet), 4)


def analyse_peak_demand(
    readings_kw: list[float],
    rate_per_kw: float = DEFAULT_DEMAND_CHARGE_RATE,
) -> PeakDemandResult:
    """Run a complete peak-demand analysis on an interval series.

    Args:
        readings_kw: Interval demand readings in kW (15-min or hourly).
        rate_per_kw: Demand charge rate in $/kW/month.

    Returns:
        Populated :class:`PeakDemandResult`.

    Raises:
        ValueError: If readings_kw is empty or contains negative values.
    """
    if not readings_kw:
        raise ValueError("readings_kw must not be empty")
    peak = peak_kw(readings_kw)
    avg = sum(readings_kw) / len(readings_kw)
    lf = load_factor(readings_kw)
    charge = demand_charge(peak, rate_per_kw)
    peak_idx = readings_kw.index(peak)
    return PeakDemandResult(
        peak_kw=peak,
        average_kw=round(avg, 6),
        load_factor=lf,
        demand_charge=charge,
        peak_index=peak_idx,
        demand_charge_rate=rate_per_kw,
    )


def demand_response_savings(
    original_peak_kw: float,
    curtailed_peak_kw: float,
    rate_per_kw: float = DEFAULT_DEMAND_CHARGE_RATE,
) -> float:
    """Estimate monthly savings from a demand-response curtailment event.

    Args:
        original_peak_kw: Peak demand before curtailment in kW.
        curtailed_peak_kw: Peak demand after curtailment in kW.
        rate_per_kw: Demand charge rate in $/kW/month.

    Returns:
        Monthly savings in dollars (non-negative).

    Raises:
        ValueError: If curtailed peak exceeds original or any value is negative.
    """
    if original_peak_kw < 0 or curtailed_peak_kw < 0:
        raise ValueError("Peak demand values must be non-negative")
    if curtailed_peak_kw > original_peak_kw:
        raise ValueError("curtailed_peak_kw cannot exceed original_peak_kw")
    reduction = original_peak_kw - curtailed_peak_kw
    return round(demand_charge(reduction, rate_per_kw), 2)
