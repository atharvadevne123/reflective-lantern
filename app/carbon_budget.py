"""Carbon budget tracking and net-zero pathway analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ScopeCategory(str, Enum):
    """GHG Protocol emission scope categories."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


@dataclass
class EmissionEntry:
    """A single emission entry in kg CO₂e."""

    category: str
    scope: ScopeCategory
    kg_co2e: float
    description: str = ""


@dataclass
class CarbonBudgetResult:
    """Result of a carbon budget analysis."""

    total_kg_co2e: float
    scope_1_kg: float
    scope_2_kg: float
    scope_3_kg: float
    budget_remaining_kg: float
    on_track: bool
    reduction_needed_pct: float


def total_emissions(entries: list[EmissionEntry]) -> float:
    """Sum all emission entries in kg CO₂e.

    Args:
        entries: List of emission entries.

    Returns:
        Total emissions in kg CO₂e.

    Raises:
        ValueError: If any entry has a negative value.
    """
    if any(e.kg_co2e < 0 for e in entries):
        raise ValueError("All emission entries must have non-negative kg_co2e")
    return round(sum(e.kg_co2e for e in entries), 4)


def emissions_by_scope(
    entries: list[EmissionEntry],
    scope: ScopeCategory,
) -> float:
    """Sum emissions for a specific GHG Protocol scope.

    Args:
        entries: List of emission entries.
        scope: Target scope category.

    Returns:
        Total emissions for the scope in kg CO₂e.
    """
    return round(sum(e.kg_co2e for e in entries if e.scope == scope), 4)


def reduction_target_kg(
    baseline_kg: float,
    target_pct: float,
) -> float:
    """Compute absolute reduction target from a percentage.

    Args:
        baseline_kg: Baseline annual emissions in kg CO₂e.
        target_pct: Reduction target as a percentage (0–100).

    Returns:
        Target absolute emissions in kg CO₂e.

    Raises:
        ValueError: If arguments are out of valid range.
    """
    if baseline_kg < 0:
        raise ValueError("baseline_kg must be non-negative")
    if not (0 <= target_pct <= 100):
        raise ValueError("target_pct must be in [0, 100]")
    return round(baseline_kg * (1.0 - target_pct / 100.0), 4)


def linear_reduction_pathway(
    baseline_kg: float,
    target_kg: float,
    years: int,
) -> list[float]:
    """Generate a linear annual reduction pathway.

    Args:
        baseline_kg: Starting emissions in kg CO₂e.
        target_kg: End-state emissions in kg CO₂e.
        years: Number of years in the pathway.

    Returns:
        List of annual emission targets (length = years + 1, including baseline year).

    Raises:
        ValueError: If arguments are invalid.
    """
    if baseline_kg < 0 or target_kg < 0:
        raise ValueError("Emissions values must be non-negative")
    if years <= 0:
        raise ValueError("years must be positive")
    step = (baseline_kg - target_kg) / years
    return [round(baseline_kg - step * y, 4) for y in range(years + 1)]


def carbon_intensity(
    total_kg_co2e: float,
    activity_unit: float,
) -> float:
    """Compute carbon intensity per unit of activity.

    Args:
        total_kg_co2e: Total emissions in kg CO₂e.
        activity_unit: Activity denominator (e.g., revenue £, floor area m², kWh).

    Returns:
        Carbon intensity in kg CO₂e per unit.

    Raises:
        ValueError: If activity_unit is non-positive or emissions are negative.
    """
    if total_kg_co2e < 0:
        raise ValueError("total_kg_co2e must be non-negative")
    if activity_unit <= 0:
        raise ValueError("activity_unit must be positive")
    return round(total_kg_co2e / activity_unit, 6)


def analyse_budget(
    entries: list[EmissionEntry],
    annual_budget_kg: float,
) -> CarbonBudgetResult:
    """Analyse emissions against a carbon budget.

    Args:
        entries: List of emission entries for the period.
        annual_budget_kg: Allowed annual carbon budget in kg CO₂e.

    Returns:
        Populated :class:`CarbonBudgetResult`.

    Raises:
        ValueError: If annual_budget_kg is negative.
    """
    if annual_budget_kg < 0:
        raise ValueError("annual_budget_kg must be non-negative")
    total = total_emissions(entries)
    s1 = emissions_by_scope(entries, ScopeCategory.SCOPE_1)
    s2 = emissions_by_scope(entries, ScopeCategory.SCOPE_2)
    s3 = emissions_by_scope(entries, ScopeCategory.SCOPE_3)
    remaining = round(annual_budget_kg - total, 4)
    on_track = total <= annual_budget_kg
    if annual_budget_kg > 0:
        reduction_needed = max(0.0, round((total - annual_budget_kg) / annual_budget_kg * 100, 2))
    else:
        reduction_needed = 0.0 if total == 0 else 100.0
    return CarbonBudgetResult(
        total_kg_co2e=total,
        scope_1_kg=s1,
        scope_2_kg=s2,
        scope_3_kg=s3,
        budget_remaining_kg=remaining,
        on_track=on_track,
        reduction_needed_pct=reduction_needed,
    )
