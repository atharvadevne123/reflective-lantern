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
        school_score * _SCHOOL_WEIGHT + transit_score * _TRANSIT_WEIGHT + walkability_score * _WALK_WEIGHT
    ) / 10.0
    risk_score = float(min(max(crime_rate, 0.0), 1.0))

    raw_score = cap_rate * _CAP_RATE_WEIGHT + amenity_composite * _AMENITY_WEIGHT - risk_score * _RISK_PENALTY
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


_SCORE_LABELS: list[tuple[float, str]] = [
    (8.0, "excellent"),
    (6.0, "good"),
    (4.0, "fair"),
    (2.0, "poor"),
]


def investment_score_label(score: float) -> str:
    """Convert a numeric investment score (0-10) to a human-readable label.

    Args:
        score: Investment score in [0, 10] as returned by
            compute_investment_analysis().

    Returns:
        One of 'excellent' (>=8), 'good' (>=6), 'fair' (>=4),
        'poor' (>=2), or 'avoid' (<2).
    """
    for threshold, label in _SCORE_LABELS:
        if score >= threshold:
            return label
    return "avoid"


def portfolio_weighted_score(
    scores: list[float],
    weights: list[float] | None = None,
) -> float:
    """Compute a weighted average investment score for a property portfolio.

    Args:
        scores: List of individual property investment scores.
        weights: Optional weight for each score (must match length). Defaults to
            equal weights.

    Returns:
        Weighted average score rounded to 3 decimal places, or 0.0 if empty.

    Raises:
        ValueError: If *weights* length doesn't match *scores* length.
    """
    if not scores:
        return 0.0
    if weights is None:
        weights = [1.0 / len(scores)] * len(scores)
    if len(weights) != len(scores):
        raise ValueError(f"weights length {len(weights)} must match scores length {len(scores)}")
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    weighted = sum(s * w for s, w in zip(scores, weights, strict=False))
    return round(weighted / total_weight, 3)


__all__ = [
    "InvestmentAnalysis",
    "compute_investment_analysis",
    "discounted_cash_flow",
    "investment_score_label",
    "irr_estimate",
    "margin_of_safety",
    "mortgage_payment",
    "payback_period",
    "portfolio_weighted_score",
    "price_to_income_ratio",
    "roi_percentage",
]


def discounted_cash_flow(
    annual_cash_flows: list[float],
    discount_rate: float,
    terminal_value: float = 0.0,
) -> float:
    """Compute the net present value (NPV) of a real estate investment.

    Discounts each year's net cash flow back to today at *discount_rate*,
    and adds a terminal (exit) value discounted to year N.

    Args:
        annual_cash_flows: Net operating income per year (NOI - debt service).
        discount_rate: Annual discount rate as a fraction (e.g. 0.08 for 8%).
        terminal_value: Expected sale proceeds at the end of the holding period.

    Returns:
        NPV in USD, rounded to 2 decimal places. Positive NPV indicates
        a value-creating investment at the given discount rate.

    Raises:
        ValueError: If *discount_rate* is negative.
    """
    if discount_rate < 0:
        raise ValueError(f"discount_rate must be non-negative, got {discount_rate}")
    if not annual_cash_flows:
        return 0.0
    n = len(annual_cash_flows)
    npv = sum(cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(annual_cash_flows))
    if terminal_value:
        npv += terminal_value / (1 + discount_rate) ** n
    return round(npv, 2)


def payback_period(
    initial_investment: float,
    annual_cash_flows: list[float],
) -> float:
    """Compute the simple payback period in years for an investment.

    Returns the (possibly fractional) year at which cumulative cash flows
    recover the initial investment. Returns float('inf') if the investment
    is never recovered within the provided cash flow horizon.

    Args:
        initial_investment: Upfront cost (positive number).
        annual_cash_flows: Net cash flows per year (can include negative years).

    Returns:
        Payback period in years; float('inf') if never recovered.

    Raises:
        ValueError: If *initial_investment* is negative.
    """
    if initial_investment < 0:
        raise ValueError(f"initial_investment must be non-negative, got {initial_investment}")
    cumulative = 0.0
    for year, cf in enumerate(annual_cash_flows, start=1):
        prev = cumulative
        cumulative += cf
        if cumulative >= initial_investment:
            remaining = initial_investment - prev
            fraction = remaining / cf if cf != 0 else 0.0
            return round(year - 1 + fraction, 4)
    return float("inf")


def margin_of_safety(
    intrinsic_value: float,
    market_price: float,
) -> float:
    """Compute the margin of safety as a percentage.

    Margin of safety = (intrinsic_value - market_price) / intrinsic_value * 100.
    A positive value indicates the asset is undervalued relative to intrinsic value.

    Args:
        intrinsic_value: Estimated intrinsic (fair) value of the property.
        market_price: Current market asking price.

    Returns:
        Margin of safety in percent, rounded to 4 decimal places.
        Returns 0.0 if intrinsic_value is zero.
    """
    if intrinsic_value == 0.0:
        return 0.0
    return round((intrinsic_value - market_price) / intrinsic_value * 100.0, 4)


def irr_estimate(
    initial_investment: float,
    annual_cash_flows: list[float],
    terminal_value: float = 0.0,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Estimate Internal Rate of Return (IRR) using bisection search.

    Finds the discount rate at which NPV = 0 for the given cash-flow stream.

    Args:
        initial_investment: Upfront cost in USD (positive number).
        annual_cash_flows: Net cash flows per year (excluding initial investment).
        terminal_value: Expected sale proceeds at end of holding period.
        iterations: Maximum bisection iterations (default 100).
        tolerance: Convergence tolerance for NPV (default 1e-6).

    Returns:
        IRR as a decimal fraction (e.g. 0.12 for 12%).
        Returns 0.0 when no positive return is possible.

    Raises:
        ValueError: If *initial_investment* is non-positive.
    """
    if initial_investment <= 0:
        raise ValueError(f"initial_investment must be positive, got {initial_investment}")
    all_flows = [-initial_investment] + list(annual_cash_flows)
    if terminal_value:
        all_flows[-1] += terminal_value

    def _npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** i for i, cf in enumerate(all_flows))

    lo, hi = -0.999, 10.0
    if _npv(lo) * _npv(hi) > 0:
        return 0.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        npv_mid = _npv(mid)
        if abs(npv_mid) < tolerance:
            return round(mid, 6)
        if _npv(lo) * npv_mid < 0:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 6)
