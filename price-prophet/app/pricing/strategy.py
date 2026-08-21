"""
Named pricing strategies for Price-Prophet.

Strategies are coarse policy decisions applied before (or instead of)
the fine-grained ML optimiser.  They encode common e-commerce pricing
playbooks as simple, auditable rules.
"""

from __future__ import annotations

import enum


class PricingStrategy(enum.Enum):
    """Enumeration of supported pricing strategies.

    Attributes
    ----------
    PENETRATION:
        Undercut the market to gain share; price is reduced.
    PREMIUM:
        Signal high quality via above-market pricing.
    COMPETITIVE:
        Track the nearest competitor slightly below their price.
    DYNAMIC:
        ML/elasticity-driven adjustment based on demand sensitivity.
    SKIMMING:
        Start high to capture early-adopter surplus; price is raised 30%.
    COST_PLUS:
        Apply a fixed 30% markup over the base/cost price.
    """

    PENETRATION = "penetration"
    PREMIUM = "premium"
    COMPETITIVE = "competitive"
    DYNAMIC = "dynamic"
    SKIMMING = "skimming"
    COST_PLUS = "cost_plus"


def apply_strategy(
    base_price: float,
    strategy: PricingStrategy,
    competitor_price: float | None = None,
    elasticity: float = -1.5,
) -> float:
    """Return a price adjusted according to the chosen *strategy*.

    Parameters
    ----------
    base_price:
        Current / reference price of the product.
    strategy:
        One of the :class:`PricingStrategy` variants.
    competitor_price:
        Best comparable competitor price.  Used only by the
        ``COMPETITIVE`` strategy; ignored by all others.
    elasticity:
        Price elasticity of demand.  Used only by the ``DYNAMIC``
        strategy to decide the direction of adjustment.

    Returns
    -------
    float
        Adjusted price, rounded to 2 decimal places.

    Notes
    -----
    * ``PENETRATION``  -> ``base_price x 0.85``
    * ``PREMIUM``      -> ``base_price x 1.25``
    * ``COMPETITIVE``  -> ``competitor_price x 0.99``
                         (falls back to *base_price* if competitor unknown)
    * ``DYNAMIC``      -> price reduced 5 % when |elasticity| > 1 (elastic);
                         price raised 5 % when demand is inelastic.
    """
    if strategy is PricingStrategy.PENETRATION:
        return round(base_price * 0.85, 2)

    if strategy is PricingStrategy.PREMIUM:
        return round(base_price * 1.25, 2)

    if strategy is PricingStrategy.COMPETITIVE:
        if competitor_price is not None and competitor_price > 0.0:
            return round(competitor_price * 0.99, 2)
        return round(base_price, 2)

    if strategy is PricingStrategy.DYNAMIC:
        if abs(elasticity) > 1.0:
            # Elastic market: lowering price grows revenue
            return round(base_price * 0.95, 2)
        else:
            # Inelastic market: raising price grows revenue
            return round(base_price * 1.05, 2)

    if strategy is PricingStrategy.SKIMMING:
        return round(base_price * 1.30, 2)

    if strategy is PricingStrategy.COST_PLUS:
        return round(base_price * 1.30, 2)

    # Fallback - return base unchanged
    return round(base_price, 2)
