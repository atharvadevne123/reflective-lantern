"""
Demand-elasticity utilities for Price-Prophet.

Price elasticity of demand (PED) measures how sensitively demand
responds to price changes.  |elasticity| > 1 → elastic (demand drops
sharply with price); |elasticity| < 1 → inelastic.
"""

from __future__ import annotations

import math
from typing import List


def estimate_elasticity(
    prices: List[float],
    demands: List[float],
) -> float:
    """Estimate the price elasticity of demand via log-log linear regression.

    Fits ``log(demand) = α + β * log(price)`` using ordinary least squares
    and returns the slope β, which is the constant-elasticity estimate.

    Pairs where price or demand are non-positive are excluded because
    the logarithm is undefined there.

    Parameters
    ----------
    prices:
        Observed price values.
    demands:
        Corresponding observed demand values.

    Returns
    -------
    float
        Estimated elasticity, or ``0.0`` if there are fewer than two
        valid data points or if the price series has zero variance.
    """
    if len(prices) != len(demands):
        return 0.0

    # Keep only valid (positive) pairs
    log_pairs = []
    for p, d in zip(prices, demands):
        if p > 0.0 and d > 0.0:
            log_pairs.append((math.log(p), math.log(d)))

    n = len(log_pairs)
    if n < 2:
        return 0.0

    log_p = [pair[0] for pair in log_pairs]
    log_d = [pair[1] for pair in log_pairs]

    mean_p = sum(log_p) / n
    mean_d = sum(log_d) / n

    ss_pp = sum((lp - mean_p) ** 2 for lp in log_p)
    ss_pd = sum((lp - mean_p) * (ld - mean_d) for lp, ld in zip(log_p, log_d))

    if ss_pp == 0.0:
        return 0.0

    return ss_pd / ss_pp


def apply_elasticity(
    base_demand: float,
    base_price: float,
    new_price: float,
    elasticity: float,
) -> float:
    """Project demand at *new_price* using the constant-elasticity model.

    Uses the formula::

        new_demand = base_demand * (new_price / base_price) ** elasticity

    Parameters
    ----------
    base_demand:
        Known demand at *base_price*.
    base_price:
        Reference price for the elasticity calculation.
    new_price:
        Price at which to project demand.
    elasticity:
        Price elasticity of demand (typically negative).

    Returns
    -------
    float
        Projected demand at *new_price*.  Returns *base_demand* unchanged
        if either price is non-positive.
    """
    if base_price <= 0.0 or new_price <= 0.0:
        return base_demand
    ratio = new_price / base_price
    return base_demand * (ratio ** elasticity)


def is_elastic(elasticity: float) -> bool:
    """Return ``True`` if demand is price-elastic (|elasticity| > 1).

    Parameters
    ----------
    elasticity:
        Estimated price elasticity of demand.

    Returns
    -------
    bool
    """
    return abs(elasticity) > 1.0
