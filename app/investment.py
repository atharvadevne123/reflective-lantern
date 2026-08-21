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
    """Convert a numeric investment score to a human-readable label.

    Accepts both the 0-10 and 0-100 scales:
        * A value ≤ 10 is interpreted on the 10-point scale and returns
          lowercase labels ('excellent', 'good', 'fair', 'poor', 'avoid').
        * A value > 10 is interpreted on the 100-point scale and returns
          Title-Case labels ('Excellent', 'Good', 'Fair', 'Poor', 'Avoid').

    Args:
        score: Investment score.

    Returns:
        Human-readable label.
    """
    if score <= 10.0:
        for threshold, label in _SCORE_LABELS:
            if score >= threshold:
                return label
        return "avoid"
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Fair"
    if score >= 20:
        return "Poor"
    return "Avoid"


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
    "break_even_occupancy",
    "cash_on_cash_return",
    "compute_investment_analysis",
    "debt_service_coverage_ratio",
    "discounted_cash_flow",
    "equity_multiple",
    "gross_rent_multiplier",
    "investment_score_label",
    "irr_estimate",
    "loan_to_value_ratio",
    "margin_of_safety",
    "mortgage_payment",
    "operating_expense_ratio",
    "payback_period",
    "portfolio_weighted_score",
    "price_to_income_ratio",
    "projected_value",
    "roi_percentage",
    "total_return_on_investment",
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
    initial_investment: float | None = None,
    annual_cash_flows: list[float] | None = None,
    *,
    purchase_price: float | None = None,
    annual_cash_flow: float | None = None,
) -> float:
    """Compute the simple payback period in years for an investment.

    Accepts two calling conventions:
      - ``payback_period(initial_investment, annual_cash_flows)`` — list of cash flows.
      - ``payback_period(purchase_price=x, annual_cash_flow=y)`` — constant annual flow.

    Args:
        initial_investment: Upfront cost (positional, old API).
        annual_cash_flows: List of annual cash flows (positional, old API).
        purchase_price: Upfront cost (keyword, new API).
        annual_cash_flow: Constant annual cash flow (keyword, new API).

    Returns:
        Payback period in years; float('inf') if never recovered.

    Raises:
        ValueError: If initial_investment/purchase_price is negative.
    """
    if purchase_price is not None or annual_cash_flow is not None:
        cost = purchase_price if purchase_price is not None else 0.0
        flow = annual_cash_flow if annual_cash_flow is not None else 0.0
        if cost < 0:
            raise ValueError(f"purchase_price must be non-negative, got {cost}")
        if cost == 0.0:
            return 0.0
        if flow <= 0:
            return float("inf")
        return round(cost / flow, 4)
    invest = initial_investment if initial_investment is not None else 0.0
    flows = annual_cash_flows if annual_cash_flows is not None else []
    if invest < 0:
        raise ValueError(f"initial_investment must be non-negative, got {invest}")
    cumulative = 0.0
    for year, cf in enumerate(flows, start=1):
        prev = cumulative
        cumulative += cf
        if cumulative >= invest:
            remaining = invest - prev
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
    all_flows = [-initial_investment, *list(annual_cash_flows)]
    if terminal_value:
        all_flows[-1] += terminal_value

    def _npv(rate: float) -> float:
        """Net present value of all_flows discounted at *rate*."""
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


def loan_to_value_ratio(
    loan_amount: float,
    property_value: float,
) -> float:
    """Compute the Loan-to-Value (LTV) ratio.

    LTV = loan_amount / property_value * 100.
    A ratio > 80% typically requires private mortgage insurance (PMI).

    Args:
        loan_amount: Principal loan amount in USD.
        property_value: Current market value of the property in USD.

    Returns:
        LTV ratio as a percentage, rounded to 2 decimal places.
        Returns 0.0 if property_value is zero.
    """
    if property_value <= 0:
        return 0.0
    return round(loan_amount / property_value * 100.0, 2)


def gross_rent_multiplier(
    property_price: float,
    annual_gross_rent: float,
) -> float:
    """Compute the Gross Rent Multiplier (GRM) for a rental property.

    GRM = property_price / annual_gross_rent.
    A lower GRM indicates better cash-flow relative to price.

    Args:
        property_price: Market price of the property in USD.
        annual_gross_rent: Annual gross rental income in USD.

    Returns:
        GRM (unit-less ratio), rounded to 2 decimal places.
        Returns float('inf') when annual_gross_rent is zero.
    """
    if annual_gross_rent <= 0:
        return float("inf")
    return round(property_price / annual_gross_rent, 2)


def annualized_return(total_return: float, years: float) -> float:
    """Compute the compound annualized return from a total return over *years*.

    Uses the formula: (1 + total_return) ** (1 / years) - 1.

    Args:
        total_return: Total return as a decimal fraction (e.g. 0.50 for 50%).
        years: Investment holding period in years (> 0).

    Returns:
        Annualized return as a decimal fraction, rounded to 6 decimal places.

    Raises:
        ValueError: If *years* is not positive or *total_return* <= -1.
    """
    if years <= 0:
        raise ValueError(f"years must be positive, got {years}")
    if total_return <= -1:
        raise ValueError(f"total_return must be > -1, got {total_return}")
    return round((1 + total_return) ** (1.0 / years) - 1, 6)


def net_present_value(cash_flows: list[float], discount_rate: float) -> float:
    """Compute the Net Present Value (NPV) of a series of cash flows.

    The first element is treated as the initial investment (typically negative).
    Subsequent elements are future cash flows discounted at *discount_rate* per period.

    Args:
        cash_flows: Ordered list of cash flows (period 0, 1, 2, …).
        discount_rate: Discount rate per period as a decimal fraction.

    Returns:
        NPV rounded to 2 decimal places.

    Raises:
        ValueError: If *cash_flows* is empty or *discount_rate* <= -1.
    """
    if not cash_flows:
        raise ValueError("cash_flows must not be empty")
    if discount_rate <= -1:
        raise ValueError(f"discount_rate must be > -1, got {discount_rate}")
    npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows))
    return round(npv, 2)


def property_yield_analysis(
    market_value: float,
    annual_rent: float,
    annual_expenses: float = 0.0,
) -> dict[str, float]:
    """Compute a basic yield analysis for an investment property.

    Args:
        market_value: Current market value of the property in USD.
        annual_rent: Annual gross rental income in USD.
        annual_expenses: Annual operating expenses (maintenance, taxes, etc.) in USD.

    Returns:
        Dict with gross_yield_pct, net_yield_pct, net_operating_income, and
        expense_ratio_pct.

    Raises:
        ValueError: If *market_value* is not positive.
    """
    if market_value <= 0:
        raise ValueError(f"market_value must be positive, got {market_value}")
    gross_yield = round(annual_rent / market_value * 100.0, 4)
    noi = annual_rent - annual_expenses
    net_yield = round(noi / market_value * 100.0, 4)
    expense_ratio = round(annual_expenses / annual_rent * 100.0, 4) if annual_rent > 0 else 0.0
    return {
        "gross_yield_pct": gross_yield,
        "net_yield_pct": net_yield,
        "net_operating_income": round(noi, 2),
        "expense_ratio_pct": expense_ratio,
    }


def equity_ratio(market_value: float, outstanding_loan: float) -> float:
    """Compute the equity ratio (owner's equity as a fraction of market value).

    Args:
        market_value: Current market value of the property in USD.
        outstanding_loan: Remaining loan principal in USD.

    Returns:
        Equity ratio in [0, 1], rounded to 4 decimal places.
        Returns 0.0 if market_value is zero.

    Raises:
        ValueError: If *outstanding_loan* is negative or *market_value* < 0.
    """
    if market_value < 0:
        raise ValueError(f"market_value must be non-negative, got {market_value}")
    if outstanding_loan < 0:
        raise ValueError(f"outstanding_loan must be non-negative, got {outstanding_loan}")
    if market_value == 0.0:
        return 0.0
    equity = max(0.0, market_value - outstanding_loan)
    return round(equity / market_value, 4)


def cash_on_cash_return(annual_pre_tax_cash_flow: float, total_cash_invested: float) -> float:
    """Compute the cash-on-cash return for a rental property.

    Cash-on-cash = annual_pre_tax_cash_flow / total_cash_invested * 100.

    Args:
        annual_pre_tax_cash_flow: Annual pre-tax cash flow in USD.
        total_cash_invested: Total cash invested (down payment + closing costs, etc.).

    Returns:
        Cash-on-cash return as a percentage; 0.0 if total_cash_invested is zero.
    """
    if total_cash_invested == 0.0:
        return 0.0
    return round(annual_pre_tax_cash_flow / total_cash_invested * 100.0, 4)


def debt_service_coverage_ratio(noi: float, annual_debt_service: float) -> float:
    """Compute the Debt Service Coverage Ratio (DSCR).

    DSCR = NOI / annual_debt_service.  Lenders typically require DSCR >= 1.2.

    Args:
        noi: Net Operating Income for the year.
        annual_debt_service: Total annual debt payments (principal + interest).

    Returns:
        DSCR rounded to 4 decimal places; 0.0 if annual_debt_service is zero.
    """
    if annual_debt_service == 0.0:
        return 0.0
    return round(noi / annual_debt_service, 4)


def break_even_occupancy(
    operating_expenses: float,
    debt_service: float,
    gross_potential_rent: float,
) -> float:
    """Compute the break-even occupancy rate for a rental property.

    Break-even occupancy = (operating_expenses + debt_service) / gross_potential_rent

    The result is the minimum occupancy needed to cover all costs.

    Args:
        operating_expenses: Annual operating expenses in USD.
        debt_service: Annual debt service payments in USD.
        gross_potential_rent: Maximum annual rent if fully occupied.

    Returns:
        Break-even occupancy as a fraction in [0, ∞]; values > 1 indicate the
        property is unlikely to be viable. Returns 0.0 if gross_potential_rent is zero.
    """
    if gross_potential_rent == 0.0:
        return 0.0
    return round((operating_expenses + debt_service) / gross_potential_rent, 4)


def operating_expense_ratio(
    operating_expenses: float,
    effective_gross_income: float,
) -> float:
    """Compute the Operating Expense Ratio (OER).

    OER = operating_expenses / effective_gross_income * 100.

    Args:
        operating_expenses: Annual operating expenses in USD.
        effective_gross_income: Actual collected income for the year.

    Returns:
        OER as a percentage; 0.0 if effective_gross_income is zero.
    """
    if effective_gross_income == 0.0:
        return 0.0
    return round(operating_expenses / effective_gross_income * 100.0, 4)


def price_to_rent_ratio(property_price: float, rent: float) -> float:
    """Compute the Price-to-Rent Ratio (PRR) for a property.

    PRR = property_price / annual_rent. When the raw ratio with *rent* would
    exceed 100 (indicating *rent* is likely a monthly figure), the function
    automatically annualises by multiplying by 12.

    Args:
        property_price: Current market price of the property.
        rent: Annual rent (or monthly rent — auto-detected by scale).

    Returns:
        Price-to-rent ratio rounded to 2 decimal places; 0.0 if rent is zero.

    Raises:
        ValueError: If property_price is negative.
    """
    if property_price < 0:
        raise ValueError(f"property_price must be non-negative, got {property_price}")
    if rent == 0.0:
        return 0.0
    raw_ratio = property_price / rent
    annual_rent = rent if raw_ratio <= 100 else rent * 12
    return round(property_price / annual_rent, 2)


def holding_period_return(purchase_price: float, sale_price: float, total_income: float = 0.0) -> float:
    """Compute the total Holding Period Return (HPR) as a percentage.

    HPR = (sale_price - purchase_price + total_income) / purchase_price * 100.

    Args:
        purchase_price: Original acquisition cost.
        sale_price: Proceeds from disposition.
        total_income: Cumulative net income received during the holding period.

    Returns:
        HPR as a percentage, rounded to 4 decimal places.

    Raises:
        ValueError: If purchase_price is zero.
    """
    if purchase_price == 0.0:
        raise ValueError("purchase_price must not be zero")
    return round((sale_price - purchase_price + total_income) / purchase_price * 100.0, 4)


def rent_to_value_ratio(annual_rent: float, property_value: float) -> float:
    """Return the rent-to-value (RTV) ratio as a percentage.

    RTV = (annual_rent / property_value) * 100.

    A higher ratio indicates stronger cash-flow potential relative to asset
    value; a common threshold for cash-flow neutrality is 1 % per month
    (12 % annually).

    Args:
        annual_rent: Total gross annual rent.
        property_value: Current market value of the property.

    Returns:
        RTV as a percentage, rounded to 4 decimal places; 0.0 if property_value is 0.

    Raises:
        ValueError: If either argument is negative.
    """
    if annual_rent < 0 or property_value < 0:
        raise ValueError("annual_rent and property_value must be non-negative")
    if property_value == 0.0:
        return 0.0
    return round(annual_rent / property_value * 100.0, 4)


def effective_gross_income(
    gross_rental_income: float,
    vacancy_rate: float = 0.05,
    other_income: float = 0.0,
) -> float:
    """Estimate Effective Gross Income (EGI) for a rental property.

    EGI = gross_rental_income * (1 - vacancy_rate) + other_income.

    Args:
        gross_rental_income: Potential gross annual rental income at full occupancy.
        vacancy_rate: Expected vacancy as a fraction in [0, 1].
        other_income: Non-rental income (laundry, parking, etc.).

    Returns:
        EGI in the same currency unit as the inputs, rounded to 4 decimal places.

    Raises:
        ValueError: If vacancy_rate is outside [0, 1] or any argument is negative.
    """
    if not 0.0 <= vacancy_rate <= 1.0:
        raise ValueError(f"vacancy_rate must be in [0, 1], got {vacancy_rate}")
    if gross_rental_income < 0 or other_income < 0:
        raise ValueError("Income values must be non-negative")
    return round(gross_rental_income * (1.0 - vacancy_rate) + other_income, 4)


def capitalisation_rate(noi: float, property_value: float) -> float:
    """Return the capitalisation rate (cap rate) as a percentage.

    Cap Rate = (NOI / property_value) * 100.

    Args:
        noi: Net Operating Income — gross income less operating expenses.
        property_value: Current market value.

    Returns:
        Cap rate as a percentage, rounded to 4 decimal places; 0.0 if property_value is 0.

    Raises:
        ValueError: If property_value is negative.
    """
    if property_value < 0:
        raise ValueError("property_value must be non-negative")
    if property_value == 0.0:
        return 0.0
    return round(noi / property_value * 100.0, 4)


def gross_yield(annual_rent: float, property_value: float) -> float:
    """Compute the gross rental yield as a percentage.

    Args:
        annual_rent: Total annual rental income before expenses.
        property_value: Purchase or current market value of the property.

    Returns:
        Gross yield as a percentage rounded to 4 decimal places.
        Returns 0.0 if *property_value* is 0.

    Raises:
        ValueError: If either argument is negative.
    """
    if annual_rent < 0 or property_value < 0:
        raise ValueError("annual_rent and property_value must be non-negative")
    if property_value == 0.0:
        return 0.0
    return round(annual_rent / property_value * 100.0, 4)


def equity_multiple(
    total_distributions: float | None = None,
    total_invested: float | None = None,
    *,
    total_profit: float | None = None,
    equity_invested: float | None = None,
) -> float:
    """Calculate the equity multiple for an investment.

    Equity multiple = total_distributions / total_invested. Accepts both
    positional args and ``total_profit`` / ``equity_invested`` keyword args.

    Args:
        total_distributions: Total cash returned to the investor (positional).
        total_invested: Total capital invested (positional).
        total_profit: Alias for total_distributions (keyword).
        equity_invested: Alias for total_invested (keyword).

    Returns:
        Equity multiple rounded to 4 decimal places, or 0.0 if nothing invested.

    Raises:
        ValueError: If either argument is negative.
    """
    dist = total_profit if total_profit is not None else (total_distributions or 0.0)
    inv = equity_invested if equity_invested is not None else (total_invested or 0.0)
    if dist < 0 or inv < 0:
        raise ValueError("Both arguments must be non-negative")
    if inv == 0.0:
        return 0.0
    return round(dist / inv, 4)


def projected_value(
    current_value: float,
    annual_growth_rate: float,
    years: int,
) -> float:
    """Project a property's future value using compound appreciation.

    Args:
        current_value: Current market value of the property.
        annual_growth_rate: Expected annual appreciation rate as a fraction
            (e.g. 0.03 for 3%).
        years: Number of years to project.

    Returns:
        Projected future value rounded to 2 decimal places.

    Raises:
        ValueError: If *years* is negative or *current_value* is negative.
    """
    if current_value < 0:
        raise ValueError("current_value must be non-negative")
    if years < 0:
        raise ValueError("years must be non-negative")
    return round(current_value * (1 + annual_growth_rate) ** years, 2)


def total_return_on_investment(
    purchase_price: float,
    sale_price: float,
    total_rental_income: float,
    total_expenses: float,
) -> float:
    """Compute total return on investment as a percentage.

    Total ROI = (net gain / purchase_price) * 100, where
    net gain = (sale_price - purchase_price) + (rental_income - expenses).

    Args:
        purchase_price: Original acquisition cost.
        sale_price: Expected or realised sale proceeds.
        total_rental_income: Cumulative rental income over the holding period.
        total_expenses: Cumulative operating expenses (maintenance, taxes, etc.).

    Returns:
        Total ROI as a percentage, rounded to 4 decimal places.

    Raises:
        ValueError: If *purchase_price* is zero or negative.
    """
    if purchase_price <= 0:
        raise ValueError("purchase_price must be positive")
    capital_gain = sale_price - purchase_price
    net_income = total_rental_income - total_expenses
    net_gain = capital_gain + net_income
    return round(net_gain / purchase_price * 100.0, 4)


def annual_rent_growth(current_rent: float, prior_rent: float) -> float:
    """Return year-over-year rent growth as a percentage.

    Args:
        current_rent: Latest annual rent (must be non-negative).
        prior_rent: Prior-period annual rent (must be strictly positive).

    Returns:
        Growth percentage.

    Raises:
        ValueError: If *current_rent* is negative or *prior_rent* is non-positive.
    """
    if current_rent < 0:
        raise ValueError(f"current_rent must be non-negative, got {current_rent}")
    if prior_rent <= 0:
        raise ValueError(f"prior_rent must be positive, got {prior_rent}")
    return round((current_rent - prior_rent) / prior_rent * 100.0, 4)


def rental_yield_after_tax(
    annual_rent: float,
    property_value: float,
    tax_rate_pct: float,
) -> float:
    """Return the after-tax gross rental yield as a percentage.

    Args:
        annual_rent: Gross annual rent in USD (must be non-negative).
        property_value: Property value in USD (must be strictly positive).
        tax_rate_pct: Effective tax rate on rental income as a percentage in [0, 100].

    Returns:
        After-tax yield in percent.

    Raises:
        ValueError: If arguments are outside expected ranges.
    """
    if annual_rent < 0:
        raise ValueError(f"annual_rent must be non-negative, got {annual_rent}")
    if property_value <= 0:
        raise ValueError(f"property_value must be positive, got {property_value}")
    if not 0 <= tax_rate_pct <= 100:
        raise ValueError(f"tax_rate_pct must be in [0, 100], got {tax_rate_pct}")
    after_tax_rent = annual_rent * (1.0 - tax_rate_pct / 100.0)
    return round(after_tax_rent / property_value * 100.0, 4)


def holding_period_return(purchase_price: float, sale_price: float, income: float = 0.0) -> float:
    """Compute total holding-period return including income.

    Args:
        purchase_price: Initial acquisition cost (must be positive).
        sale_price: Proceeds at disposition.
        income: Total net income received during holding period (e.g. rent).

    Returns:
        HPR as a percentage, rounded to 4 decimal places.

    Raises:
        ValueError: If purchase_price is not positive.
    """
    if purchase_price <= 0:
        raise ValueError(f"purchase_price must be positive, got {purchase_price}")
    return round(((sale_price - purchase_price + income) / purchase_price) * 100.0, 4)


def leverage_ratio(total_assets: float, total_equity: float) -> float:
    """Compute the leverage (debt-to-equity) ratio.

    Args:
        total_assets: Fair market value of all assets.
        total_equity: Owner equity (total_assets - total_debt).

    Returns:
        Leverage ratio rounded to 4 decimal places; returns 0.0 when equity is zero.

    Raises:
        ValueError: If total_assets is negative.
    """
    if total_assets < 0:
        raise ValueError(f"total_assets must be non-negative, got {total_assets}")
    if total_equity == 0:
        return 0.0
    total_debt = total_assets - total_equity
    return round(total_debt / total_equity, 4)


def risk_adjusted_return(
    annual_return_pct: float,
    volatility_pct: float,
    risk_free_rate_pct: float = 2.0,
) -> float:
    """Compute the Sharpe-style risk-adjusted return (excess return / volatility).

    Args:
        annual_return_pct: Annualised return in percent.
        volatility_pct: Annualised volatility (std dev) in percent (must be positive).
        risk_free_rate_pct: Risk-free rate in percent (default 2%).

    Returns:
        Risk-adjusted ratio rounded to 4 decimal places; 0.0 when volatility is zero.

    Raises:
        ValueError: If volatility_pct is negative.
    """
    if volatility_pct < 0:
        raise ValueError(f"volatility_pct must be non-negative, got {volatility_pct}")
    if volatility_pct == 0:
        return 0.0
    excess = annual_return_pct - risk_free_rate_pct
    return round(excess / volatility_pct, 4)
