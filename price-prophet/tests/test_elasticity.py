"""Tests for app/pricing/elasticity.py."""

from __future__ import annotations

import pytest


def test_estimate_elasticity_negative():
    from app.pricing.elasticity import estimate_elasticity

    prices = [10.0, 12.0, 14.0, 16.0]
    demands = [200.0, 170.0, 145.0, 120.0]
    elast = estimate_elasticity(prices, demands)
    assert elast < 0


def test_estimate_elasticity_too_short():
    from app.pricing.elasticity import estimate_elasticity

    assert estimate_elasticity([10.0], [100.0]) == 0.0


def test_estimate_elasticity_mismatched():
    from app.pricing.elasticity import estimate_elasticity

    assert estimate_elasticity([1.0, 2.0], [1.0]) == 0.0


def test_apply_elasticity_higher_price():
    from app.pricing.elasticity import apply_elasticity

    base_demand = 100.0
    new_demand = apply_elasticity(base_demand, 10.0, 12.0, -1.5)
    assert new_demand < base_demand


def test_apply_elasticity_zero_base_price():
    from app.pricing.elasticity import apply_elasticity

    assert apply_elasticity(100.0, 0.0, 10.0, -1.5) == 100.0


def test_is_elastic_true():
    from app.pricing.elasticity import is_elastic

    assert is_elastic(-2.0) is True


def test_is_elastic_false():
    from app.pricing.elasticity import is_elastic

    assert is_elastic(-0.5) is False


@pytest.mark.parametrize(
    "elasticity,expected", [(-2.0, True), (-1.0, False), (-0.5, False), (-1.001, True)]
)
def test_is_elastic_parametrized(elasticity: float, expected: bool) -> None:
    from app.pricing.elasticity import is_elastic

    assert is_elastic(elasticity) is expected


@pytest.mark.parametrize(
    "base_demand,base_price,new_price,elasticity",
    [
        (100.0, 10.0, 12.0, -1.5),
        (200.0, 20.0, 22.0, -2.0),
        (50.0, 5.0, 4.0, -0.8),
    ],
)
def test_apply_elasticity_finite_result(
    base_demand: float, base_price: float, new_price: float, elasticity: float
) -> None:
    from app.pricing.elasticity import apply_elasticity

    result = apply_elasticity(base_demand, base_price, new_price, elasticity)
    assert result >= 0.0
    assert result != float("inf")


class TestRevenueAtPrice:
    def test_base_price_equals_base_revenue(self) -> None:
        from app.pricing.elasticity import revenue_at_price

        result = revenue_at_price(10.0, 100.0, 10.0, -1.0)
        assert result == pytest.approx(1000.0, rel=1e-3)

    def test_zero_price_returns_zero(self) -> None:
        from app.pricing.elasticity import revenue_at_price

        assert revenue_at_price(0.0, 100.0, 10.0, -1.0) == 0.0

    def test_higher_price_lower_demand_elastic(self) -> None:
        from app.pricing.elasticity import revenue_at_price

        rev_base = revenue_at_price(10.0, 100.0, 10.0, -2.0)
        rev_high = revenue_at_price(15.0, 100.0, 10.0, -2.0)
        assert rev_high < rev_base

    def test_returns_float(self) -> None:
        from app.pricing.elasticity import revenue_at_price

        result = revenue_at_price(10.0, 100.0, 10.0, -1.0)
        assert isinstance(result, float)


class TestOptimalPriceDirection:
    def test_elastic_demand_lower(self) -> None:
        from app.pricing.elasticity import optimal_price_direction

        assert optimal_price_direction(10.0, -2.0) == "lower"

    def test_inelastic_demand_raise(self) -> None:
        from app.pricing.elasticity import optimal_price_direction

        assert optimal_price_direction(10.0, -0.5) == "raise"

    def test_unit_elastic_hold(self) -> None:
        from app.pricing.elasticity import optimal_price_direction

        assert optimal_price_direction(10.0, -1.0) == "hold"

    @pytest.mark.parametrize("e,expected", [(-3.0, "lower"), (-0.2, "raise"), (-1.0, "hold")])
    def test_parametrized(self, e: float, expected: str) -> None:
        from app.pricing.elasticity import optimal_price_direction

        assert optimal_price_direction(5.0, e) == expected


def test_estimate_elasticity_identical_prices_returns_zero() -> None:
    from app.pricing.elasticity import estimate_elasticity

    result = estimate_elasticity([10.0, 10.0, 10.0], [100.0, 200.0, 150.0])
    assert result == 0.0


def test_estimate_elasticity_fewer_than_two_valid_returns_zero() -> None:
    from app.pricing.elasticity import estimate_elasticity

    result = estimate_elasticity([0.0], [100.0])
    assert result == 0.0


def test_apply_elasticity_same_price_returns_base_demand() -> None:
    from app.pricing.elasticity import apply_elasticity

    result = apply_elasticity(200.0, 50.0, 50.0, -1.5)
    assert result == pytest.approx(200.0, abs=1e-6)


@pytest.mark.parametrize("elasticity", [-0.5, -1.0, -2.0, -3.0])
def test_is_elastic_threshold_boundary(elasticity: float) -> None:
    from app.pricing.elasticity import is_elastic

    expected = abs(elasticity) > 1.0
    assert is_elastic(elasticity) is expected


@pytest.mark.parametrize("base_demand,price_from,price_to", [(100.0, 50.0, 60.0), (200.0, 100.0, 80.0)])
def test_apply_elasticity_returns_non_negative(base_demand: float, price_from: float, price_to: float) -> None:
    from app.pricing.elasticity import apply_elasticity

    result = apply_elasticity(base_demand, price_from, price_to, -1.0)
    assert result >= 0.0


def test_estimate_elasticity_matching_lists_returns_float() -> None:
    from app.pricing.elasticity import estimate_elasticity

    prices = [10.0, 12.0, 15.0, 20.0]
    demands = [200.0, 180.0, 150.0, 100.0]
    result = estimate_elasticity(prices, demands)
    assert isinstance(result, float)
