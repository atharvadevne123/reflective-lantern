"""Investment scoring and property analytics for Realty-Edge.

Computes a 0-10 investment score from cap rate, amenity quality, and
risk factors. Also provides rental yield estimation and break-even
horizon calculation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_CAP_RATE_WEIGHT = 100.0
_AMENITY_WEIGHT = 5.0
_RISK_PENALTY = 5.0

DEFAULT_OPERATING_EXPENSE_RATIO = 0.35
_SCHOOL_WEIGHT = 0.4
_TRANSIT_WEIGHT = 0.3
_WALK_WEIGHT = 0.3
INVESTMENT_SCORE_MAX = 10.0
INVESTMENT_SCORE_MIN = 0.0


@dataclass
class InvestmentAnalysis:
    """Full investment analysis for a single property."""

    predicted_value: float
    annual_rent_estimate: float
    gross_rental_yield: float
    cap_rate: float
    investment_score: float
    break_even_years: float
    amenity_composite: float
    risk_score: float


def compute_investment_analysis(
    predicted_value: float,
    avg_rental_yield: float,
    school_score: float,
    transit_score: float,
    walkability_score: float,
    crime_rate: float,
    operating_expense_ratio: float = DEFAULT_OPERATING_EXPENSE_RATIO,
) -> InvestmentAnalysis:
    """Compute a full investment analysis for a property.

    Args:
        predicted_value: Estimated market value in USD.
        avg_rental_yield: Gross annual rental yield as a fraction (e.g. 0.06).
        school_score: School quality score 0-10.
        transit_score: Transit quality score 0-10.
        walkability_score: Walkability score 0-10.
        crime_rate: Normalised crime rate 0-1 (higher = more crime).
        operating_expense_ratio: Fraction of gross rent consumed by expenses.

    Returns:
        InvestmentAnalysis dataclass with all computed metrics.
    """
    if predicted_value <= 0:
        return InvestmentAnalysis(
            predicted_value=predicted_value,
            annual_rent_estimate=0.0,
            gross_rental_yield=0.0,
            cap_rate=0.0,
            investment_score=0.0,
            break_even_years=float("inf"),
            amenity_composite=0.0,
            risk_score=crime_rate,
        )

    annual_rent = predicted_value * avg_rental_yield
    noi = annual_rent * (1.0 - operating_expense_ratio)
    cap_rate = noi / predicted_value

    amenity_composite = (
        school_score * _SCHOOL_WEIGHT
        + transit_score * _TRANSIT_WEIGHT
        + walkability_score * _WALK_WEIGHT
    ) / 10.0
    risk_score = float(min(max(crime_rate, 0.0), 1.0))

    raw_score = (
        cap_rate * _CAP_RATE_WEIGHT
        + amenity_composite * _AMENITY_WEIGHT
        - risk_score * _RISK_PENALTY
    )
    investment_score = float(min(max(raw_score, INVESTMENT_SCORE_MIN), INVESTMENT_SCORE_MAX))

    break_even = predicted_value / noi if noi > 0 else float("inf")

    logger.debug(
        "InvestmentAnalysis value=%.0f rent=%.0f cap_rate=%.4f score=%.2f",
        predicted_value,
        annual_rent,
        cap_rate,
        investment_score,
    )

    return InvestmentAnalysis(
        predicted_value=predicted_value,
        annual_rent_estimate=round(annual_rent, 2),
        gross_rental_yield=round(avg_rental_yield, 4),
        cap_rate=round(cap_rate, 4),
        investment_score=round(investment_score, 2),
        break_even_years=round(break_even, 1),
        amenity_composite=round(amenity_composite, 4),
        risk_score=round(risk_score, 4),
    )


def mortgage_payment(
    principal: float,
    annual_rate: float,
    term_years: int = 30,
    down_payment_pct: float = 0.20,
) -> float:
    """Compute the fixed monthly mortgage payment.

    Args:
        principal: Property purchase price in USD.
        annual_rate: Annual interest rate as a fraction (e.g. 0.065 for 6.5%).
        term_years: Loan term in years (default 30).
        down_payment_pct: Down payment as a fraction of principal (default 0.20).

    Returns:
        Monthly payment in USD, rounded to 2 decimal places.
        Returns 0.0 when the interest rate is zero (simple division).
    """
    loan = principal * (1.0 - down_payment_pct)
    if loan <= 0:
        return 0.0
    monthly_rate = annual_rate / 12.0
    n = term_years * 12
    if monthly_rate == 0.0:
        return round(loan / n, 2)
    payment = loan * (monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)
    return round(payment, 2)


def roi_percentage(
    predicted_value: float,
    purchase_price: float,
    annual_income: float,
    annual_expenses: float,
    hold_years: int = 5,
) -> float:
    """Compute the total return on investment over *hold_years*.

    Args:
        predicted_value: Estimated exit value of the property in USD.
        purchase_price: Initial acquisition cost in USD.
        annual_income: Annual gross rental income in USD.
        annual_expenses: Annual total expenses (mortgage, maintenance, taxes) in USD.
        hold_years: Number of years the property is held.

    Returns:
        Annualised ROI as a percentage (e.g. 8.5 means 8.5 %).
        Returns 0.0 when purchase_price is zero or negative.
    """
    if purchase_price <= 0:
        return 0.0
    net_operating_income = (annual_income - annual_expenses) * hold_years
    appreciation = predicted_value - purchase_price
    total_return = net_operating_income + appreciation
    annualised = (total_return / purchase_price) / max(hold_years, 1) * 100.0
    return round(annualised, 2)


def price_to_income_ratio(property_price: float, annual_household_income: float) -> float:
    """Return the price-to-income ratio for affordability assessment.

    Args:
        property_price: Property market value in USD.
        annual_household_income: Buyer's gross annual household income in USD.

    Returns:
        Price-to-income ratio, or ``math.inf`` when income is zero.
    """
    if annual_household_income <= 0:
        return math.inf
    return round(property_price / annual_household_income, 2)
