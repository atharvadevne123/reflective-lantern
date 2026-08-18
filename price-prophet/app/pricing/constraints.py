"""
Hard pricing constraints for Price-Prophet.

Constraints act as guardrails applied *after* the optimiser or strategy
layer has produced a candidate price, ensuring business rules are never
violated regardless of model output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PricingConstraints:
    """Hard limits applied to every computed price.

    Attributes
    ----------
    min_price:
        Absolute floor - no price may go below this value (default 0.01).
    max_price:
        Absolute ceiling - no price may exceed this value (default 99999.0).
    max_change_pct:
        Maximum allowed percentage change from the *base_price*, in
        either direction (default 50.0 → ±50 %).
    """

    min_price: float = 0.01
    max_price: float = 99_999.0
    max_change_pct: float = 50.0


def apply_constraints(
    price: float,
    base_price: float,
    constraints: PricingConstraints,
) -> float:
    """Clip *price* so that it satisfies all *constraints*.

    The order of operations is:

    1. Clip the candidate price to the
       ``[base_price ± max_change_pct]`` band.
    2. Clip the result to ``[min_price, max_price]``.

    Parameters
    ----------
    price:
        Candidate price produced by the optimiser or strategy layer.
    base_price:
        Current / reference price of the product.
    constraints:
        Constraint bounds to enforce.

    Returns
    -------
    float
        Adjusted price that satisfies all constraints.
    """
    # 1. Apply max-change constraint relative to base_price
    if base_price > 0.0:
        max_delta = base_price * (constraints.max_change_pct / 100.0)
        price = max(base_price - max_delta, min(base_price + max_delta, price))

    # 2. Apply absolute floor / ceiling
    price = max(constraints.min_price, min(constraints.max_price, price))

    return price


def violates_constraints(
    price: float,
    base_price: float,
    constraints: PricingConstraints,
) -> bool:
    """Return ``True`` if *price* violates any of *constraints*.

    Parameters
    ----------
    price:
        Price to check.
    base_price:
        Current / reference price of the product.
    constraints:
        Constraint bounds to check against.

    Returns
    -------
    bool
        ``True`` if the price would need to be adjusted, ``False``
        if it already satisfies all constraints.
    """
    if price < constraints.min_price or price > constraints.max_price:
        return True

    if base_price > 0.0:
        change_pct = abs(price - base_price) / base_price * 100.0
        if change_pct > constraints.max_change_pct:
            return True

    return False


def constraint_headroom(price: float, constraints: PricingConstraints) -> dict[str, float]:
    """Compute how much room the price has before hitting constraint boundaries.

    Args:
        price: Current price to evaluate.
        constraints: Active constraint bounds.

    Returns:
        Dict with 'room_to_min' (price - min_price, floored at 0) and
        'room_to_max' (max_price - price, floored at 0).
    """
    return {
        "room_to_min": round(max(0.0, price - constraints.min_price), 4),
        "room_to_max": round(max(0.0, constraints.max_price - price), 4),
    }


def batch_apply_constraints(
    prices: list[float],
    constraints: PricingConstraints,
    base_price: float = 0.0,
) -> list[float]:
    """Apply constraints to a list of candidate prices.

    Args:
        prices: Candidate prices to constrain.
        constraints: Constraint bounds.
        base_price: Reference price for max-change-pct guard.

    Returns:
        List of constrained prices in the same order.
    """
    return [apply_constraints(p, base_price, constraints) for p in prices]
