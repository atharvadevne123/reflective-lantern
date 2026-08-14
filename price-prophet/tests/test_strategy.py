"""Tests for app/pricing/strategy.py."""

from __future__ import annotations

import pytest


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


@pytest.mark.parametrize(
    "strategy_name,expected_direction",
    [
        ("PENETRATION", "lower"),
        ("SKIMMING", "higher"),
    ],
)
def test_strategy_direction_parametrized(strategy_name: str, expected_direction: str) -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    strategy = getattr(PricingStrategy, strategy_name)
    price = apply_strategy(100.0, strategy)
    if expected_direction == "lower":
        assert price < 100.0
    else:
        assert price > 100.0


@pytest.mark.parametrize("base_price", [10.0, 50.0, 100.0, 500.0])
def test_cost_plus_strategy_output_positive(base_price: float) -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    price = apply_strategy(base_price, PricingStrategy.COST_PLUS)
    assert price > 0.0


def test_strategy_result_is_float() -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    price = apply_strategy(100.0, PricingStrategy.PENETRATION)
    assert isinstance(price, float)
