"""Energy efficiency investment ROI and payback period calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ROIResult:
    """Return on investment analysis for an energy efficiency measure."""

    capex: float
    annual_savings: float
    simple_payback_years: float
    irr_percent: float | None
    npv: float
    lifetime_savings: float
    roi_percent: float


def simple_payback(capex: float, annual_savings: float) -> float:
    """Compute simple payback period in years.

    Args:
        capex: Upfront capital expenditure.
        annual_savings: Annual financial savings (revenue or cost avoided).

    Returns:
        Payback in years, or infinity if annual_savings is zero.

    Raises:
        ValueError: If capex or annual_savings is negative.
    """
    if capex < 0 or annual_savings < 0:
        raise ValueError("capex and annual_savings must be non-negative")
    if annual_savings == 0.0:
        return float("inf")
    return round(capex / annual_savings, 4)


def net_present_value(
    capex: float,
    annual_cashflows: list[float],
    discount_rate: float,
) -> float:
    """Compute Net Present Value (NPV) of a project.

    Args:
        capex: Initial investment (positive number; sign is reversed internally).
        annual_cashflows: Cash flows for each year, starting year 1.
        discount_rate: Annual discount rate as a decimal (e.g. 0.08 = 8%).

    Returns:
        NPV in currency units.

    Raises:
        ValueError: If discount_rate <= -1 or capex is negative.
    """
    if capex < 0:
        raise ValueError("capex must be non-negative")
    if discount_rate <= -1:
        raise ValueError("discount_rate must be > -1")
    npv = -capex
    for t, cf in enumerate(annual_cashflows, start=1):
        npv += cf / (1 + discount_rate) ** t
    return round(npv, 4)


def lifetime_savings(
    annual_savings: float,
    lifetime_years: int,
    escalation_rate: float = 0.03,
) -> float:
    """Compute cumulative nominal lifetime savings with annual escalation.

    Args:
        annual_savings: Year-1 savings in currency units.
        lifetime_years: Project lifetime in years.
        escalation_rate: Annual energy price escalation rate (default 3%).

    Returns:
        Total nominal savings over the lifetime.

    Raises:
        ValueError: If lifetime_years < 1 or annual_savings is negative.
    """
    if lifetime_years < 1:
        raise ValueError("lifetime_years must be >= 1")
    if annual_savings < 0:
        raise ValueError("annual_savings must be non-negative")
    total = 0.0
    for year in range(lifetime_years):
        total += annual_savings * (1 + escalation_rate) ** year
    return round(total, 4)


def roi_percent(capex: float, total_savings: float) -> float:
    """Compute simple ROI as (total_savings - capex) / capex * 100.

    Args:
        capex: Capital expenditure.
        total_savings: Total lifetime (or period) savings.

    Returns:
        ROI percentage.  Returns -100.0 when capex is zero and savings are zero.

    Raises:
        ValueError: If capex is negative.
    """
    if capex < 0:
        raise ValueError("capex must be non-negative")
    if capex == 0.0:
        return 0.0
    return round((total_savings - capex) / capex * 100.0, 4)


def analyse_investment(
    capex: float,
    annual_savings: float,
    lifetime_years: int = 20,
    discount_rate: float = 0.08,
    escalation_rate: float = 0.03,
) -> ROIResult:
    """Full ROI analysis for an energy efficiency capital project.

    Args:
        capex: Upfront capital cost.
        annual_savings: Year-1 annual financial savings.
        lifetime_years: Project lifetime in years (default 20).
        discount_rate: Annual discount rate for NPV (default 8%).
        escalation_rate: Annual savings escalation rate (default 3%).

    Returns:
        Populated :class:`ROIResult`.
    """
    payback = simple_payback(capex, annual_savings)
    cashflows = [annual_savings * (1 + escalation_rate) ** y for y in range(lifetime_years)]
    npv = net_present_value(capex, cashflows, discount_rate)
    total = lifetime_savings(annual_savings, lifetime_years, escalation_rate)
    roi = roi_percent(capex, total)
    return ROIResult(
        capex=capex,
        annual_savings=annual_savings,
        simple_payback_years=payback,
        irr_percent=None,
        npv=npv,
        lifetime_savings=total,
        roi_percent=roi,
    )


def energy_cost_savings(
    kwh_saved_per_year: float,
    energy_rate: float,
    o_and_m_savings: float = 0.0,
) -> float:
    """Compute annual financial savings from energy reduction.

    Args:
        kwh_saved_per_year: Annual energy reduction in kWh.
        energy_rate: Energy tariff in $/kWh.
        o_and_m_savings: Additional O&M savings per year.

    Returns:
        Total annual savings in currency units.

    Raises:
        ValueError: If any argument is negative.
    """
    if kwh_saved_per_year < 0 or energy_rate < 0 or o_and_m_savings < 0:
        raise ValueError("All arguments must be non-negative")
    return round(kwh_saved_per_year * energy_rate + o_and_m_savings, 4)
