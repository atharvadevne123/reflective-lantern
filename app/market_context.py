"""Market context enrichment for property valuations.

Computes relative pricing metrics that give buyers and investors context
beyond the raw predicted value: price per square foot percentile, days-on-
market classification, affordability index, and price-to-rent ratio.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DOM_FAST_DAYS = 14
DOM_SLOW_DAYS = 60
AFFORDABILITY_THRESHOLD = 28.0
STRETCHED_THRESHOLD = 36.0
DEFAULT_DOWN_PAYMENT = 0.20
DEFAULT_MORTGAGE_RATE = 0.065
DEFAULT_TERM_YEARS = 30
DEFAULT_ANNUAL_INCOME = 100_000.0


def price_per_sqft(predicted_value: float, sqft: float) -> float:
    """Return price per square foot, or 0 if sqft is non-positive."""
    if sqft <= 0:
        return 0.0
    return round(predicted_value / sqft, 2)


def dom_classification(listing_days: int) -> str:
    """Classify days-on-market as a string bucket.

    Args:
        listing_days: Number of days the property has been on market.

    Returns:
        One of 'fast' (<14), 'normal' (14-60), 'slow' (>60).
    """
    if listing_days < DOM_FAST_DAYS:
        return "fast"
    if listing_days <= DOM_SLOW_DAYS:
        return "normal"
    return "slow"


def affordability_index(
    predicted_value: float,
    annual_income: float = DEFAULT_ANNUAL_INCOME,
    down_payment_pct: float = DEFAULT_DOWN_PAYMENT,
    mortgage_rate: float = DEFAULT_MORTGAGE_RATE,
    term_years: int = DEFAULT_TERM_YEARS,
) -> dict[str, Any]:
    """Compute a mortgage affordability index.

    Args:
        predicted_value: Property price in USD.
        annual_income: Annual household income.
        down_payment_pct: Down payment as a fraction of purchase price.
        mortgage_rate: Annual mortgage interest rate.
        term_years: Loan term in years.

    Returns:
        Dict with loan_amount, monthly_payment, pct_income, is_affordable.
    """
    loan = predicted_value * (1 - down_payment_pct)
    r = mortgage_rate / 12
    n = term_years * 12
    monthly = loan * r * (1 + r) ** n / ((1 + r) ** n - 1) if r > 0 else loan / n
    pct_income = monthly / (annual_income / 12) * 100
    is_affordable = pct_income <= AFFORDABILITY_THRESHOLD
    logger.debug(
        "Affordability: monthly=%.0f pct_income=%.1f%% affordable=%s",
        monthly,
        pct_income,
        is_affordable,
    )
    return {
        "loan_amount": round(loan, 2),
        "monthly_payment": round(monthly, 2),
        "pct_income": round(pct_income, 2),
        "is_affordable": bool(is_affordable),
    }


def price_to_rent_ratio(predicted_value: float, annual_rent: float) -> float:
    """Return the price-to-rent ratio (lower = better for buyers).

    A ratio below 15 typically favours buying; above 20 favours renting.
    """
    if annual_rent <= 0:
        return float("inf")
    return round(predicted_value / annual_rent, 2)


def affordability_bucket(pct_income: float) -> str:
    """Classify a mortgage payment as a percentage of income into a bucket.

    Args:
        pct_income: Monthly mortgage payment as a percentage of monthly income.

    Returns:
        'affordable' (<=28%), 'stretched' (28-36%), or 'unaffordable' (>36%).
    """
    if pct_income <= AFFORDABILITY_THRESHOLD:
        return "affordable"
    if pct_income <= STRETCHED_THRESHOLD:
        return "stretched"
    return "unaffordable"


def rent_vs_buy_comparison(
    predicted_value: float,
    annual_rent: float,
    annual_income: float = DEFAULT_ANNUAL_INCOME,
    mortgage_rate: float = DEFAULT_MORTGAGE_RATE,
    term_years: int = DEFAULT_TERM_YEARS,
    down_payment_pct: float = DEFAULT_DOWN_PAYMENT,
) -> dict[str, Any]:
    """Compare the financial cost of renting vs buying for a property.

    Args:
        predicted_value: Property purchase price in USD.
        annual_rent: Annual rent for a comparable property in USD.
        annual_income: Buyer's annual household income in USD.
        mortgage_rate: Annual mortgage interest rate as a fraction.
        term_years: Mortgage term in years.
        down_payment_pct: Down payment as a fraction of the purchase price.

    Returns:
        Dict with monthly_rent, monthly_mortgage, rent_premium, recommendation,
        and price_to_rent_ratio.
    """
    loan = predicted_value * (1 - down_payment_pct)
    r = mortgage_rate / 12
    n = term_years * 12
    monthly_mortgage = loan * r * (1 + r) ** n / ((1 + r) ** n - 1) if r > 0 else loan / n
    monthly_rent = annual_rent / 12
    rent_premium = monthly_rent - monthly_mortgage
    ptr = price_to_rent_ratio(predicted_value, annual_rent)
    recommendation = "buy" if ptr < 15 else ("neutral" if ptr <= 20 else "rent")
    return {
        "monthly_rent": round(monthly_rent, 2),
        "monthly_mortgage": round(monthly_mortgage, 2),
        "rent_premium": round(rent_premium, 2),
        "price_to_rent_ratio": ptr,
        "recommendation": recommendation,
    }


def price_trend_indicator(
    current_price: float,
    previous_price: float,
    periods: int = 1,
) -> dict[str, Any]:
    """Compute price change direction and magnitude over *periods*.

    Args:
        current_price: Most recent property value in USD.
        previous_price: Prior property value in USD (must be positive).
        periods: Number of periods elapsed (used for annualised rate).

    Returns:
        Dict with absolute_change, pct_change, annualised_pct_change, and trend.
    """
    if previous_price <= 0:
        return {
            "absolute_change": 0.0,
            "pct_change": 0.0,
            "annualised_pct_change": 0.0,
            "trend": "unknown",
        }
    abs_change = current_price - previous_price
    pct_change = abs_change / previous_price * 100.0
    ann = (pct_change / max(periods, 1)) if periods else pct_change
    if pct_change > 2.0:
        trend = "rising"
    elif pct_change < -2.0:
        trend = "falling"
    else:
        trend = "stable"
    return {
        "absolute_change": round(abs_change, 2),
        "pct_change": round(pct_change, 2),
        "annualised_pct_change": round(ann, 2),
        "trend": trend,
    }


def affordability_ratio(
    predicted_value: float,
    annual_income: float,
    down_payment_pct: float = DEFAULT_DOWN_PAYMENT,
) -> float:
    """Return the price-to-income ratio after down payment.

    A common rule-of-thumb benchmark: values ≤ 3.0 are considered affordable,
    3.0-5.0 moderately stretched, and >5.0 unaffordable.

    Args:
        predicted_value: Property price in USD.
        annual_income: Annual household income in USD.
        down_payment_pct: Down payment fraction (reduces the financed amount).

    Returns:
        Ratio of financed amount to annual income (0.0 if income is zero).
    """
    if annual_income <= 0:
        return 0.0
    financed = predicted_value * (1.0 - down_payment_pct)
    return round(financed / annual_income, 3)


def housing_affordability_index(
    median_home_price: float,
    median_household_income: float,
    mortgage_rate: float = 0.065,
    term_years: int = 30,
    down_payment_pct: float = 0.20,
) -> dict[str, float]:
    """Calculate a Housing Affordability Index (HAI).

    HAI = (qualifying income / actual income) * 100.  Values above 100 mean
    the median household can afford the median home.

    Args:
        median_home_price: Median home price in USD.
        median_household_income: Median annual household income in USD.
        mortgage_rate: Annual mortgage interest rate as a decimal (default 6.5%).
        term_years: Loan amortisation term in years (default 30).
        down_payment_pct: Down payment fraction (default 20%).

    Returns:
        Dict with loan_amount, monthly_payment, qualifying_income, and hai.
    """
    if median_household_income <= 0 or median_home_price <= 0:
        return {
            "loan_amount": 0.0,
            "monthly_payment": 0.0,
            "qualifying_income": 0.0,
            "hai": 0.0,
        }
    loan = median_home_price * (1.0 - down_payment_pct)
    n = term_years * 12
    if mortgage_rate <= 0:
        monthly_payment = loan / n
    else:
        r = mortgage_rate / 12
        monthly_payment = loan * r / (1 - (1 + r) ** -n)
    qualifying_income = monthly_payment * 12 / 0.28  # 28% front-end ratio rule
    hai = (median_household_income / qualifying_income) * 100 if qualifying_income > 0 else 0.0
    return {
        "loan_amount": round(loan, 2),
        "monthly_payment": round(monthly_payment, 2),
        "qualifying_income": round(qualifying_income, 2),
        "hai": round(hai, 2),
    }


def price_trend_consistency(
    price_series: list[float],
) -> dict[str, object]:
    """Analyse price trend consistency across a series of periodic price snapshots.

    Args:
        price_series: Ordered list of property values (at least 2 values).

    Returns:
        Dict with 'direction_changes', 'consistency_score' (0-1), and 'trend'.
        'trend' is 'rising', 'falling', or 'mixed'.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    if len(price_series) < 2:
        raise ValueError("price_series must have at least 2 elements")
    deltas = [price_series[i + 1] - price_series[i] for i in range(len(price_series) - 1)]
    rises = sum(1 for d in deltas if d > 0)
    falls = sum(1 for d in deltas if d < 0)
    direction_changes = sum(1 for i in range(len(deltas) - 1) if (deltas[i] > 0) != (deltas[i + 1] > 0))
    total = len(deltas)
    majority = max(rises, falls)
    consistency_score = round(majority / total, 4) if total else 0.0
    if rises > falls:
        trend = "rising"
    elif falls > rises:
        trend = "falling"
    else:
        trend = "mixed"
    return {
        "direction_changes": direction_changes,
        "consistency_score": consistency_score,
        "trend": trend,
        "periods": total,
    }


def market_summary(
    predicted_value: float,
    annual_rent: float,
    listing_days: int,
    sqft: float,
    annual_income: float = DEFAULT_ANNUAL_INCOME,
) -> dict[str, object]:
    """Return a single-call market context summary for a property.

    Combines price-per-sqft, DOM classification, affordability, and
    price-to-rent ratio into one convenience dict.

    Args:
        predicted_value: Property price in USD.
        annual_rent: Estimated annual rent in USD.
        listing_days: Number of days the property has been on market.
        sqft: Property size in square feet.
        annual_income: Annual household income for affordability calculation.

    Returns:
        Dict with keys: price_per_sqft, dom, dom_classification,
        price_to_rent_ratio, affordability.
    """
    affordability = affordability_index(predicted_value, annual_income=annual_income)
    return {
        "price_per_sqft": price_per_sqft(predicted_value, sqft),
        "dom": listing_days,
        "dom_classification": dom_classification(listing_days),
        "price_to_rent_ratio": price_to_rent_ratio(predicted_value, annual_rent),
        "affordability": affordability,
    }

__all__ = [
    "affordability_bucket",
    "affordability_index",
    "affordability_ratio",
    "comparable_value_adjustment",
    "dom_classification",
    "effective_gross_income",
    "housing_affordability_index",
    "market_heat_score",
    "market_summary",
    "price_appreciation_rate",
    "price_per_bedroom",
    "price_per_sqft",
    "price_to_rent_ratio",
    "price_trend_consistency",
    "price_trend_indicator",
    "rent_vs_buy_comparison",
    "value_gap",
]


def price_per_bedroom(predicted_value: float, bedrooms: int) -> float:
    """Return price per bedroom, or 0 if bedrooms is non-positive.

    Args:
        predicted_value: Estimated property price in USD.
        bedrooms: Number of bedrooms.

    Returns:
        Price per bedroom rounded to 2 decimal places; 0.0 if bedrooms <= 0.
    """
    if bedrooms <= 0:
        return 0.0
    return round(predicted_value / bedrooms, 2)


def market_heat_score(
    days_on_market: int,
    list_to_sale_ratio: float,
    inventory_months: float,
) -> float:
    """Compute a composite market heat score in [0, 10].

    Higher scores indicate a hotter (seller's) market.

    Args:
        days_on_market: Median days on market for recent sales.
        list_to_sale_ratio: Ratio of sale price to list price (e.g. 1.02 = 2% over).
        inventory_months: Months of housing supply currently available.

    Returns:
        Heat score in [0, 10], rounded to 2 decimal places.
    """
    dom_score = max(0.0, 1.0 - days_on_market / 90.0) * 10.0
    ratio_score = min(10.0, max(0.0, (list_to_sale_ratio - 0.9) * 100.0))
    inv_score = max(0.0, 1.0 - inventory_months / 6.0) * 10.0
    composite = (dom_score + ratio_score + inv_score) / 3.0
    return round(min(10.0, max(0.0, composite)), 2)


def value_gap(estimated_value: float, list_price: float) -> float:
    """Compute the gap between estimated value and list price as a percentage.

    A positive gap means the property is listed below estimated value (underpriced).

    Args:
        estimated_value: Model-estimated fair market value in USD.
        list_price: Actual listing price in USD.

    Returns:
        Gap percentage rounded to 4 decimal places; 0.0 if list_price is zero.
    """
    if list_price == 0.0:
        return 0.0
    return round((estimated_value - list_price) / list_price * 100.0, 4)


def comparable_value_adjustment(
    subject_sqft: float,
    comp_sqft: float,
    comp_price: float,
    price_per_sqft_adj: float = 100.0,
) -> float:
    """Estimate a value adjustment for a subject property relative to a comparable.

    Adjusts *comp_price* upward or downward by the per-sqft value of the
    square-footage difference.

    Args:
        subject_sqft: Square footage of the subject property.
        comp_sqft: Square footage of the comparable sale.
        comp_price: Sale price of the comparable property in USD.
        price_per_sqft_adj: Dollar adjustment per sqft difference.

    Returns:
        Adjusted value for the subject property, rounded to 2 decimal places.
    """
    diff = subject_sqft - comp_sqft
    return round(comp_price + diff * price_per_sqft_adj, 2)


def effective_gross_income(
    potential_gross_rent: float,
    vacancy_rate: float = 0.05,
) -> float:
    """Compute Effective Gross Income (EGI) after applying a vacancy allowance.

    EGI = potential_gross_rent * (1 - vacancy_rate)

    Args:
        potential_gross_rent: Gross rental income if fully occupied (USD/year).
        vacancy_rate: Expected vacancy as a fraction in [0, 1] (default 5%).

    Returns:
        EGI in USD/year, rounded to 2 decimal places.

    Raises:
        ValueError: If *vacancy_rate* is not in [0, 1].
    """
    if not (0.0 <= vacancy_rate <= 1.0):
        raise ValueError(f"vacancy_rate must be in [0, 1], got {vacancy_rate}")
    return round(potential_gross_rent * (1.0 - vacancy_rate), 2)


def price_appreciation_rate(
    original_price: float,
    current_price: float,
    years: float,
) -> float:
    """Compute the annualised price appreciation rate (CAGR) for a property.

    Args:
        original_price: Purchase price in USD (must be > 0).
        current_price: Current market value in USD (must be > 0).
        years: Number of years between purchase and valuation (must be > 0).

    Returns:
        Annualised appreciation rate as a percentage, rounded to 4 decimal places.

    Raises:
        ValueError: If any argument is non-positive.
    """

    if original_price <= 0 or current_price <= 0 or years <= 0:
        raise ValueError("original_price, current_price, and years must all be positive")
    cagr = (current_price / original_price) ** (1.0 / years) - 1.0
    return round(cagr * 100.0, 4)
