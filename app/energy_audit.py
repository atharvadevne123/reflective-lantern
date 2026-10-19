"""Building energy audit utilities and ASHRAE Level I/II assessment helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AuditLevel(str, Enum):
    """ASHRAE energy audit levels."""

    LEVEL_1 = "walk_through"
    LEVEL_2 = "energy_survey"
    LEVEL_3 = "investment_grade"


class EndUseCategory(str, Enum):
    """Building end-use categories."""

    HVAC = "hvac"
    LIGHTING = "lighting"
    PLUG_LOADS = "plug_loads"
    HOT_WATER = "hot_water"
    PROCESS = "process"
    OTHER = "other"


@dataclass
class EndUseBreakdown:
    """Energy consumption broken down by end use."""

    category: EndUseCategory
    annual_kwh: float
    annual_cost: float
    pct_of_total: float


@dataclass
class AuditOpportunity:
    """An identified energy saving opportunity."""

    description: str
    category: EndUseCategory
    estimated_annual_savings_kwh: float
    estimated_annual_savings_cost: float
    implementation_cost: float
    simple_payback_years: float
    priority: str


@dataclass
class AuditReport:
    """Complete energy audit report."""

    building_id: str
    audit_level: AuditLevel
    total_annual_kwh: float
    total_annual_cost: float
    energy_use_intensity_kwh_per_m2: float
    end_use_breakdown: list[EndUseBreakdown]
    opportunities: list[AuditOpportunity]
    total_savings_potential_kwh: float
    total_savings_potential_cost: float


def energy_use_intensity(
    annual_kwh: float,
    floor_area_m2: float,
) -> float:
    """Compute building Energy Use Intensity (EUI).

    Args:
        annual_kwh: Total annual energy consumption in kWh.
        floor_area_m2: Conditioned floor area in m².

    Returns:
        EUI in kWh/m².

    Raises:
        ValueError: If arguments are non-positive.
    """
    if annual_kwh < 0:
        raise ValueError("annual_kwh must be non-negative")
    if floor_area_m2 <= 0:
        raise ValueError("floor_area_m2 must be positive")
    return round(annual_kwh / floor_area_m2, 4)


def end_use_percentage(
    end_use_kwh: float,
    total_kwh: float,
) -> float:
    """Compute end-use share of total energy consumption.

    Args:
        end_use_kwh: Energy consumed by end use in kWh.
        total_kwh: Total building consumption in kWh.

    Returns:
        Percentage of total consumption.

    Raises:
        ValueError: If total_kwh is non-positive.
    """
    if total_kwh <= 0:
        raise ValueError("total_kwh must be positive")
    if end_use_kwh < 0:
        raise ValueError("end_use_kwh must be non-negative")
    return round(end_use_kwh / total_kwh * 100.0, 4)


def opportunity_payback(
    annual_savings: float,
    implementation_cost: float,
) -> float:
    """Compute simple payback period for an energy measure.

    Args:
        annual_savings: Annual cost savings in currency units.
        implementation_cost: Total implementation cost in currency units.

    Returns:
        Simple payback period in years.

    Raises:
        ValueError: If annual_savings is non-positive or implementation_cost is negative.
    """
    if annual_savings <= 0:
        raise ValueError("annual_savings must be positive")
    if implementation_cost < 0:
        raise ValueError("implementation_cost must be non-negative")
    return round(implementation_cost / annual_savings, 4)


def prioritise_opportunities(
    opportunities: list[AuditOpportunity],
) -> list[AuditOpportunity]:
    """Sort opportunities by payback period (shortest first).

    Args:
        opportunities: List of audit opportunities.

    Returns:
        Sorted list of opportunities.
    """
    return sorted(opportunities, key=lambda o: o.simple_payback_years)


def build_end_use_breakdown(
    end_uses: dict[EndUseCategory, float],
    energy_rate: float,
) -> list[EndUseBreakdown]:
    """Build a list of end-use breakdowns from a consumption dictionary.

    Args:
        end_uses: Mapping of end-use category to annual kWh.
        energy_rate: Electricity unit rate in currency/kWh.

    Returns:
        List of :class:`EndUseBreakdown` objects.

    Raises:
        ValueError: If any consumption value is negative or energy_rate is non-positive.
    """
    if any(v < 0 for v in end_uses.values()):
        raise ValueError("All consumption values must be non-negative")
    if energy_rate <= 0:
        raise ValueError("energy_rate must be positive")
    total_kwh = sum(end_uses.values())
    result = []
    for category, kwh in end_uses.items():
        pct = end_use_percentage(kwh, total_kwh) if total_kwh > 0 else 0.0
        result.append(EndUseBreakdown(
            category=category,
            annual_kwh=round(kwh, 4),
            annual_cost=round(kwh * energy_rate, 2),
            pct_of_total=pct,
        ))
    return result


def run_audit(
    building_id: str,
    floor_area_m2: float,
    end_uses: dict[EndUseCategory, float],
    energy_rate: float,
    opportunities: list[AuditOpportunity],
    audit_level: AuditLevel = AuditLevel.LEVEL_2,
) -> AuditReport:
    """Run a building energy audit and produce a report.

    Args:
        building_id: Unique building identifier.
        floor_area_m2: Conditioned floor area in m².
        end_uses: Annual consumption per end-use category in kWh.
        energy_rate: Electricity unit rate in currency/kWh.
        opportunities: List of pre-identified saving opportunities.
        audit_level: ASHRAE audit level classification.

    Returns:
        Populated :class:`AuditReport`.
    """
    total_kwh = sum(end_uses.values())
    total_cost = round(total_kwh * energy_rate, 2)
    eui = energy_use_intensity(total_kwh, floor_area_m2)
    breakdown = build_end_use_breakdown(end_uses, energy_rate)
    sorted_ops = prioritise_opportunities(opportunities)
    savings_kwh = round(sum(o.estimated_annual_savings_kwh for o in opportunities), 4)
    savings_cost = round(sum(o.estimated_annual_savings_cost for o in opportunities), 2)
    return AuditReport(
        building_id=building_id,
        audit_level=audit_level,
        total_annual_kwh=round(total_kwh, 4),
        total_annual_cost=total_cost,
        energy_use_intensity_kwh_per_m2=eui,
        end_use_breakdown=breakdown,
        opportunities=sorted_ops,
        total_savings_potential_kwh=savings_kwh,
        total_savings_potential_cost=savings_cost,
    )
