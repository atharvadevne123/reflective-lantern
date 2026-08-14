"""
Feature engineering for Price-Prophet.

Functions here derive pricing-specific features from raw record fields.
They all operate on plain Python types (float, dict, list) so no ML
framework is required at the feature-construction stage.
"""

from __future__ import annotations

from typing import Any


def price_ratio(price: float, reference: float) -> float:
    """Compute the ratio of *price* to *reference*.

    Parameters
    ----------
    price:
        The price to evaluate.
    reference:
        The reference (e.g. base or average) price.

    Returns
    -------
    float
        ``price / reference``, or ``1.0`` when *reference* is zero to
        avoid division-by-zero.
    """
    if reference == 0.0:
        return 1.0
    return price / reference


def demand_elasticity_feature(
    demand: float,
    price: float,
    base_price: float,
) -> float:
    """Estimate point-in-time price elasticity of demand.

    Computed as::

        (demand_change_pct / price_change_pct)

    where *demand* is treated as the observed value and ``1.0`` (unit
    demand) as the baseline, giving ``(demand - 1) / demand`` as a
    proxy for the demand-change percentage.

    Returns ``0.0`` when *price* equals *base_price* (no price change)
    to avoid division-by-zero.

    Parameters
    ----------
    demand:
        Observed demand at *price*.
    price:
        Observed price.
    base_price:
        Reference / baseline price.

    Returns
    -------
    float
        Elasticity estimate, or ``0.0`` if there is no price change.
    """
    if price == base_price or base_price == 0.0:
        return 0.0

    price_change_pct = (price - base_price) / base_price
    # Treat base demand as 1.0 (unit-normalised)
    demand_change_pct = (demand - 1.0) / max(demand, 1e-9)
    return demand_change_pct / price_change_pct


def competitive_gap(own_price: float, competitor_price: float) -> float:
    """Return the relative price gap versus a competitor.

    A positive result means *own_price* is above *competitor_price*
    (potentially losing sales); a negative result means it is below.

    Parameters
    ----------
    own_price:
        The product's own current price.
    competitor_price:
        The best comparable competitor price.

    Returns
    -------
    float
        ``(own_price - competitor_price) / competitor_price``, or
        ``0.0`` when *competitor_price* is zero or negative.
    """
    if competitor_price <= 0.0:
        return 0.0
    return (own_price - competitor_price) / competitor_price


def build_feature_matrix(records: list[dict[str, Any]]) -> list[list[float]]:
    """Convert a list of record dicts into a numeric feature matrix.

    Each row in the returned matrix corresponds to one record and
    contains the following features in order:

    0. ``base_price``
    1. ``demand``
    2. ``competition_price``
    3. ``day_of_week``
    4. ``is_weekend`` (cast to float: 0.0 or 1.0)

    Missing fields default to ``0.0``.

    Parameters
    ----------
    records:
        Raw data records as returned by the loader or validator.

    Returns
    -------
    list[list[float]]
        2-D feature matrix; one inner list per record.
    """
    matrix: list[list[float]] = []

    for rec in records:
        base_price = float(rec.get("base_price", 0.0))
        demand = float(rec.get("demand", 0.0))
        competition_price = float(rec.get("competition_price", 0.0))
        day_of_week = float(rec.get("day_of_week", 0.0))
        is_weekend = float(bool(rec.get("is_weekend", False)))

        matrix.append([base_price, demand, competition_price, day_of_week, is_weekend])

    return matrix
