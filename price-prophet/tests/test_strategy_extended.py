"""Extended pricing strategy tests with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.pricing.strategy import PricingStrategy, apply_strategy


def test_penetration_lowers_price() -> None:
    result = apply_strategy(100.0, PricingStrategy.PENETRATION)
    assert result < 100.0
    assert result == pytest.approx(85.0)


def test_premium_raises_price() -> None:
    result = apply_strategy(100.0, PricingStrategy.PREMIUM)
    assert result > 100.0
    assert result == pytest.approx(125.0)


def test_skimming_raises_price() -> None:
    result = apply_strategy(100.0, PricingStrategy.SKIMMING)
    assert result == pytest.approx(130.0)


def test_cost_plus_applies_markup() -> None:
    result = apply_strategy(100.0, PricingStrategy.COST_PLUS)
    assert result == pytest.approx(130.0)


def test_competitive_uses_competitor_price() -> None:
    result = apply_strategy(100.0, PricingStrategy.COMPETITIVE, competitor_price=90.0)
    assert result == pytest.approx(90.0 * 0.99)


def test_competitive_fallback_to_base_when_no_competitor() -> None:
    result = apply_strategy(100.0, PricingStrategy.COMPETITIVE, competitor_price=None)
    assert result == pytest.approx(100.0)


def test_competitive_fallback_when_competitor_zero() -> None:
    result = apply_strategy(100.0, PricingStrategy.COMPETITIVE, competitor_price=0.0)
    assert result == pytest.approx(100.0)


def test_dynamic_elastic_market_lowers_price() -> None:
    result = apply_strategy(100.0, PricingStrategy.DYNAMIC, elasticity=-2.0)
    assert result < 100.0
    assert result == pytest.approx(95.0)


def test_dynamic_inelastic_market_raises_price() -> None:
    result = apply_strategy(100.0, PricingStrategy.DYNAMIC, elasticity=-0.5)
    assert result > 100.0
    assert result == pytest.approx(105.0)


@pytest.mark.parametrize(
    "strategy,base_price",
    [
        (PricingStrategy.PENETRATION, 200.0),
        (PricingStrategy.PREMIUM, 200.0),
        (PricingStrategy.SKIMMING, 200.0),
        (PricingStrategy.COST_PLUS, 200.0),
    ],
)
def test_strategy_scales_with_base_price(strategy: PricingStrategy, base_price: float) -> None:
    result_100 = apply_strategy(100.0, strategy)
    result_200 = apply_strategy(base_price, strategy)
    assert result_200 == pytest.approx(result_100 * 2, abs=0.01)


@pytest.mark.parametrize(
    "elasticity,expected_direction",
    [
        (-2.0, "lower"),
        (-1.5, "lower"),
        (-0.5, "higher"),
        (-0.9, "higher"),
    ],
)
def test_dynamic_direction_by_elasticity(elasticity: float, expected_direction: str) -> None:
    base = 100.0
    result = apply_strategy(base, PricingStrategy.DYNAMIC, elasticity=elasticity)
    if expected_direction == "lower":
        assert result < base
    else:
        assert result > base


def test_all_strategies_return_positive_price() -> None:
    for strategy in PricingStrategy:
        result = apply_strategy(100.0, strategy, competitor_price=90.0)
        assert result > 0.0


def test_result_rounded_to_two_decimal_places() -> None:
    result = apply_strategy(99.99, PricingStrategy.PREMIUM)
    assert result == round(result, 2)


def test_pricing_strategy_enum_members() -> None:
    names = {s.value for s in PricingStrategy}
    assert "penetration" in names
    assert "premium" in names
    assert "dynamic" in names
    assert "competitive" in names
    assert "skimming" in names
    assert "cost_plus" in names
