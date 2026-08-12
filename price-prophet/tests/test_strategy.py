"""Tests for app/pricing/strategy.py."""
from __future__ import annotations


def test_penetration_strategy():
    from app.pricing.strategy import PricingStrategy, apply_strategy
    price = apply_strategy(100.0, PricingStrategy.PENETRATION)
    assert abs(price - 85.0) < 0.01


def test_premium_strategy():
    from app.pricing.strategy import PricingStrategy, apply_strategy
    price = apply_strategy(100.0, PricingStrategy.PREMIUM)
    assert abs(price - 125.0) < 0.01


def test_competitive_strategy_with_competitor():
    from app.pricing.strategy import PricingStrategy, apply_strategy
    price = apply_strategy(100.0, PricingStrategy.COMPETITIVE, competitor_price=90.0)
    assert abs(price - 89.10) < 0.01


def test_competitive_strategy_no_competitor():
    from app.pricing.strategy import PricingStrategy, apply_strategy
    price = apply_strategy(100.0, PricingStrategy.COMPETITIVE, competitor_price=None)
    assert price == 100.0


def test_dynamic_strategy_elastic():
    from app.pricing.strategy import PricingStrategy, apply_strategy
    price = apply_strategy(100.0, PricingStrategy.DYNAMIC, elasticity=-2.0)
    assert abs(price - 95.0) < 0.01


def test_dynamic_strategy_inelastic():
    from app.pricing.strategy import PricingStrategy, apply_strategy
    price = apply_strategy(100.0, PricingStrategy.DYNAMIC, elasticity=-0.5)
    assert abs(price - 105.0) < 0.01
