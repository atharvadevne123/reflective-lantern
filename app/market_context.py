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


def neighbourhood_score(
    school_score: float,
    transit_score: float,
    walkability_score: float,
    safety_score: float = 7.0,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute a composite neighbourhood quality score (0-10).

    Each component is weighted and combined into a single index.  Scores
    outside [0, 10] are clamped before aggregation.

    Args:
        school_score: School quality rating 0-10.
        transit_score: Public transit quality 0-10.
        walkability_score: Walk Score equivalent 0-10.
        safety_score: Neighbourhood safety score 0-10 (default 7.0).
        weights: Custom weight mapping with keys 'school', 'transit',
            'walkability', 'safety'.  Must sum to 1.0; defaults to
            equal weighting (0.25 each).

    Returns:
        Dict with 'composite_score' (rounded to 2 dp), 'grade' (A-D),
        and individual component scores.

    Raises:
        ValueError: If custom *weights* do not sum to approximately 1.0.
    """
    default_w = {"school": 0.25, "transit": 0.25, "walkability": 0.25, "safety": 0.25}
    w = weights if weights is not None else default_w
    if weights is not None:
        total_w = sum(w.values())
        if abs(total_w - 1.0) > 0.01:
            raise ValueError(f"weights must sum to 1.0, got {total_w:.4f}")
    components = {
        "school": max(0.0, min(10.0, school_score)),
        "transit": max(0.0, min(10.0, transit_score)),
        "walkability": max(0.0, min(10.0, walkability_score)),
        "safety": max(0.0, min(10.0, safety_score)),
    }
    composite = sum(components[k] * w.get(k, 0.0) for k in components)
    grade = "A" if composite >= 8.0 else "B" if composite >= 6.0 else "C" if composite >= 4.0 else "D"
    return {
        "composite_score": round(composite, 2),
        "grade": grade,
        **{f"{k}_score": v for k, v in components.items()},
    }


__all__ = [
    "affordability_bucket",
    "affordability_index",
    "affordability_ratio",
    "dom_classification",
    "housing_affordability_index",
    "market_summary",
    "neighbourhood_score",
    "price_per_sqft",
    "price_to_rent_ratio",
    "price_trend_consistency",
    "price_trend_indicator",
    "rent_vs_buy_comparison",
]


def price_growth_rate(old_price: float, new_price: float, years: float = 1.0) -> float:
    """Compute the annualized price growth rate between two values.

    Args:
        old_price: Starting price in USD.
        new_price: Ending price in USD.
        years: Number of years between measurements (> 0).

    Returns:
        Annualized growth rate as a percentage, rounded to 4 decimal places.

    Raises:
        ValueError: If *old_price* is not positive or *years* is not positive.
    """
    if old_price <= 0:
        raise ValueError(f"old_price must be positive, got {old_price}")
    if years <= 0:
        raise ValueError(f"years must be positive, got {years}")
    rate = ((new_price / old_price) ** (1.0 / years) - 1.0) * 100.0
    return round(rate, 4)


def market_heat_index(dom: int, pct_over_asking: float) -> str:
    """Classify market temperature based on days-on-market and over-asking percentage.

    A simple heuristic that classifies the market as 'hot', 'warm', 'neutral',
    or 'cool' based on two key signals.

    Args:
        dom: Median days on market for recently sold properties.
        pct_over_asking: Percentage of homes selling above asking price (0-100).

    Returns:
        One of 'hot', 'warm', 'neutral', or 'cool'.
    """
    if dom < 14 and pct_over_asking >= 50:
        return "hot"
    if dom < 30 and pct_over_asking >= 25:
        return "warm"
    if dom < 60:
        return "neutral"
    return "cool"


def vacancy_adjusted_yield(gross_yield: float, vacancy_rate: float) -> float:
    """Compute the vacancy-adjusted net yield for a rental property.

    Accounts for the proportion of the year the property sits vacant.

    Args:
        gross_yield: Annual gross rental yield as a percentage (e.g. 6.0 for 6%).
        vacancy_rate: Expected vacancy rate as a decimal fraction (e.g. 0.05 for 5%).

    Returns:
        Effective yield after accounting for vacancy, rounded to 4 decimal places.

    Raises:
        ValueError: If *vacancy_rate* is not in [0, 1].
    """
    if not (0.0 <= vacancy_rate <= 1.0):
        raise ValueError(f"vacancy_rate must be in [0, 1], got {vacancy_rate}")
    return round(gross_yield * (1.0 - vacancy_rate), 4)


def normalize_market_features(features: dict[str, float]) -> dict[str, float]:
    """Normalize a dict of market feature values to [0, 1] range.

    Uses min-max normalization across all values in the dict.

    Args:
        features: Dict of feature_name -> numeric value.

    Returns:
        Dict with the same keys; values normalized to [0, 1].
        If all values are equal, all normalized values are 0.5.

    Raises:
        ValueError: If *features* is empty.
    """
    if not features:
        raise ValueError("features must not be empty")
    min_v = min(features.values())
    max_v = max(features.values())
    rng = max_v - min_v
    if rng < 1e-12:
        return {k: 0.5 for k in features}
    return {k: round((v - min_v) / rng, 6) for k, v in features.items()}
