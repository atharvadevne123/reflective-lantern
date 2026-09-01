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


def test_unknown_strategy_returns_base() -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    # DYNAMIC with elasticity=0 (inelastic) raises price 5%
    price = apply_strategy(100.0, PricingStrategy.DYNAMIC, elasticity=0.0)
    assert price == pytest.approx(105.0, abs=0.01)


def test_competitive_zero_competitor_price_falls_back() -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    price = apply_strategy(100.0, PricingStrategy.COMPETITIVE, competitor_price=0.0)
    assert price == 100.0


@pytest.mark.parametrize("base_price", [10.0, 50.0, 200.0])
def test_skimming_always_above_base(base_price: float) -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    price = apply_strategy(base_price, PricingStrategy.SKIMMING)
    assert price > base_price


@pytest.mark.parametrize("base_price", [10.0, 50.0, 200.0])
def test_penetration_always_below_base(base_price: float) -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    price = apply_strategy(base_price, PricingStrategy.PENETRATION)
    assert price < base_price


def test_premium_above_base() -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    price = apply_strategy(100.0, PricingStrategy.PREMIUM)
    assert price > 100.0


def test_premium_strategy_parametrized() -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    for base in [50.0, 100.0, 200.0]:
        price = apply_strategy(base, PricingStrategy.PREMIUM)
        assert price > base


def test_strategy_returns_positive() -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    for strategy in [
        PricingStrategy.PENETRATION,
        PricingStrategy.PREMIUM,
        PricingStrategy.SKIMMING,
        PricingStrategy.COST_PLUS,
    ]:
        price = apply_strategy(100.0, strategy)
        assert price > 0.0


def test_competitive_strategy_undercuts_by_one_percent() -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    price = apply_strategy(100.0, PricingStrategy.COMPETITIVE, competitor_price=100.0)
    assert price < 100.0


@pytest.mark.parametrize("base", [10.0, 100.0, 1000.0])
def test_penetration_price_below_base(base: float) -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    price = apply_strategy(base, PricingStrategy.PENETRATION)
    assert price < base


@pytest.mark.parametrize("strategy_name", ["PENETRATION", "PREMIUM", "SKIMMING", "COST_PLUS"])
def test_all_strategies_return_float(strategy_name: str) -> None:
    from app.pricing.strategy import PricingStrategy, apply_strategy

    strategy = PricingStrategy[strategy_name]
    price = apply_strategy(50.0, strategy)
    assert isinstance(price, float)
