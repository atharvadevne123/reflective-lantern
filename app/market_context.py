"""Market context enrichment for property valuations.

Computes relative pricing metrics that give buyers and investors context
beyond the raw predicted value: price per square foot percentile, days-on-
market classification, affordability index, and price-to-rent ratio.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
    if listing_days < 14:
        return "fast"
    if listing_days <= 60:
        return "normal"
    return "slow"


def affordability_index(
    predicted_value: float,
    annual_income: float = 100_000.0,
    down_payment_pct: float = 0.20,
    mortgage_rate: float = 0.065,
    term_years: int = 30,
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
    is_affordable = pct_income <= 28.0
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
    if pct_income <= 28.0:
        return "affordable"
    if pct_income <= 36.0:
        return "stretched"
    return "unaffordable"
