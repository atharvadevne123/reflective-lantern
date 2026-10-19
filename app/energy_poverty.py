"""Energy poverty and fuel poverty assessment tools for household analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

UK_10_PERCENT_THRESHOLD: float = 0.10
"""UK legacy 10% rule: household spending ≥10% of income on energy."""

LIHC_LOW_INCOME_THRESHOLD: float = 0.60
"""Low Income High Costs: income below 60% of median."""


class PovertyMethod(str, Enum):
    """Fuel poverty assessment methodology."""

    TEN_PERCENT = "10_percent"
    LIHC = "lihc"
    AFTER_FUEL_COSTS = "after_fuel_costs"


@dataclass
class FuelPovertyAssessment:
    """Assessment result for a single household."""

    household_id: str
    annual_energy_cost: float
    annual_income: float
    energy_cost_ratio: float
    is_fuel_poor: bool
    method: PovertyMethod
    vulnerability_score: float
    recommended_interventions: list[str]


def energy_cost_ratio(
    annual_energy_cost: float,
    annual_income: float,
) -> float:
    """Compute the fraction of income spent on energy.

    Args:
        annual_energy_cost: Annual household energy expenditure.
        annual_income: Gross annual household income.

    Returns:
        Ratio of energy cost to income (0 = no cost, 1 = all income).

    Raises:
        ValueError: If either argument is negative, or income is zero.
    """
    if annual_energy_cost < 0 or annual_income < 0:
        raise ValueError("Values must be non-negative")
    if annual_income == 0.0:
        raise ValueError("annual_income must be positive")
    return round(annual_energy_cost / annual_income, 6)


def assess_ten_percent(
    annual_energy_cost: float,
    annual_income: float,
) -> bool:
    """Assess fuel poverty using the UK legacy 10% rule.

    Args:
        annual_energy_cost: Annual energy expenditure.
        annual_income: Gross annual income.

    Returns:
        True if energy costs represent ≥10% of income.
    """
    return energy_cost_ratio(annual_energy_cost, annual_income) >= UK_10_PERCENT_THRESHOLD


def after_fuel_costs_income(
    annual_income: float,
    annual_energy_cost: float,
) -> float:
    """Compute residual income after deducting energy costs.

    Args:
        annual_income: Gross annual household income.
        annual_energy_cost: Annual energy expenditure.

    Returns:
        Residual income (may be negative if costs exceed income).

    Raises:
        ValueError: If either argument is negative.
    """
    if annual_income < 0 or annual_energy_cost < 0:
        raise ValueError("Values must be non-negative")
    return round(annual_income - annual_energy_cost, 4)


def vulnerability_score(
    age_household_head: int,
    has_disability: bool,
    epc_rating: str,
    children_under_5: int = 0,
) -> float:
    """Compute a simple vulnerability score for a fuel-poor household.

    Args:
        age_household_head: Age of the household head in years.
        has_disability: Whether any household member has a disability.
        epc_rating: Energy Performance Certificate label (A-G).
        children_under_5: Number of children under 5 years old.

    Returns:
        Vulnerability score in [0, 100].

    Raises:
        ValueError: If epc_rating is not in A-G.
    """
    epc_rating = epc_rating.upper()
    if epc_rating not in "ABCDEFG" or len(epc_rating) != 1:
        raise ValueError("epc_rating must be A-G")
    score = 0.0
    if age_household_head >= 75:
        score += 30.0
    elif age_household_head >= 65:
        score += 15.0
    if has_disability:
        score += 20.0
    epc_scores = {"A": 0, "B": 5, "C": 10, "D": 20, "E": 30, "F": 40, "G": 50}
    score += epc_scores[epc_rating]
    score += min(10.0, children_under_5 * 5.0)
    return round(min(100.0, score), 2)


def recommended_interventions(
    epc_rating: str,
    is_fuel_poor: bool,
    has_disability: bool,
) -> list[str]:
    """Generate a prioritised list of support interventions.

    Args:
        epc_rating: Current EPC label (A-G).
        is_fuel_poor: Whether the household is assessed as fuel poor.
        has_disability: Presence of disability in household.

    Returns:
        List of recommended intervention strings.
    """
    recs: list[str] = []
    if is_fuel_poor:
        recs.append("Apply for Warm Home Discount or similar fuel poverty scheme")
    epc_rating = epc_rating.upper()
    if epc_rating in ("D", "E", "F", "G"):
        recs.append("Refer to Energy Company Obligation (ECO) for insulation grants")
    if epc_rating in ("F", "G"):
        recs.append("Prioritise fabric-first measures: loft and cavity wall insulation")
    if has_disability:
        recs.append("Priority Services Register eligibility — contact energy supplier")
    recs.append("Conduct detailed in-home energy assessment")
    return recs


def assess_household(
    household_id: str,
    annual_energy_cost: float,
    annual_income: float,
    age_head: int,
    has_disability: bool,
    epc_rating: str,
    children_under_5: int = 0,
    method: PovertyMethod = PovertyMethod.TEN_PERCENT,
) -> FuelPovertyAssessment:
    """Run a complete fuel poverty assessment for a household.

    Args:
        household_id: Unique household identifier.
        annual_energy_cost: Annual energy expenditure.
        annual_income: Gross annual income.
        age_head: Age of household head.
        has_disability: Disability flag.
        epc_rating: EPC label.
        children_under_5: Number of young children.
        method: Assessment methodology.

    Returns:
        Populated :class:`FuelPovertyAssessment`.
    """
    ratio = energy_cost_ratio(annual_energy_cost, annual_income)
    if method == PovertyMethod.TEN_PERCENT:
        poor = assess_ten_percent(annual_energy_cost, annual_income)
    else:
        poor = ratio >= UK_10_PERCENT_THRESHOLD
    vscore = vulnerability_score(age_head, has_disability, epc_rating, children_under_5)
    interventions = recommended_interventions(epc_rating, poor, has_disability)
    return FuelPovertyAssessment(
        household_id=household_id,
        annual_energy_cost=annual_energy_cost,
        annual_income=annual_income,
        energy_cost_ratio=ratio,
        is_fuel_poor=poor,
        method=method,
        vulnerability_score=vscore,
        recommended_interventions=interventions,
    )
