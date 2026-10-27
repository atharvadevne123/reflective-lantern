"""Extended tests for price-prophet app/pricing/elasticity.py."""

from __future__ import annotations

import pytest
from app.pricing.elasticity import (
    apply_elasticity,
    estimate_elasticity,
    is_elastic,
    optimal_price_direction,
    revenue_at_price,
)


class TestEstimateElasticity:
    def test_returns_float(self) -> None:
        result = estimate_elasticity([1.0, 2.0, 3.0], [100.0, 80.0, 60.0])
        assert isinstance(result, float)

    def test_mismatched_lengths_returns_zero(self) -> None:
        assert estimate_elasticity([1.0, 2.0], [100.0]) == pytest.approx(0.0)

    def test_fewer_than_two_valid_returns_zero(self) -> None:
        assert estimate_elasticity([1.0], [100.0]) == pytest.approx(0.0)

    def test_non_positive_prices_excluded(self) -> None:
        result = estimate_elasticity([0.0, -1.0, 2.0, 3.0], [100.0, 90.0, 80.0, 60.0])
        assert isinstance(result, float)

    def test_all_invalid_returns_zero(self) -> None:
        assert estimate_elasticity([0.0, -1.0], [100.0, 90.0]) == pytest.approx(0.0)

    def test_constant_price_returns_zero(self) -> None:
        assert estimate_elasticity([5.0, 5.0, 5.0], [100.0, 90.0, 80.0]) == pytest.approx(0.0)

    def test_negative_elasticity_for_normal_good(self) -> None:
        # demand falls as price rises
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        demands = [100.0, 70.0, 50.0, 35.0, 25.0]
        result = estimate_elasticity(prices, demands)
        assert result < 0


class TestApplyElasticity:
    def test_same_price_returns_base_demand(self) -> None:
        result = apply_elasticity(100.0, 10.0, 10.0, -1.5)
        assert result == pytest.approx(100.0)

    def test_zero_base_price_returns_base_demand(self) -> None:
        result = apply_elasticity(100.0, 0.0, 10.0, -1.5)
        assert result == pytest.approx(100.0)

    def test_zero_new_price_returns_base_demand(self) -> None:
        result = apply_elasticity(100.0, 10.0, 0.0, -1.5)
        assert result == pytest.approx(100.0)

    def test_price_increase_reduces_demand_for_negative_elasticity(self) -> None:
        result = apply_elasticity(100.0, 10.0, 15.0, -1.5)
        assert result < 100.0

    def test_price_decrease_increases_demand_for_negative_elasticity(self) -> None:
        result = apply_elasticity(100.0, 10.0, 5.0, -1.5)
        assert result > 100.0


class TestIsElastic:
    def test_elastic_when_abs_gt_one(self) -> None:
        assert is_elastic(-1.5) is True
        assert is_elastic(2.0) is True

    def test_inelastic_when_abs_lt_one(self) -> None:
        assert is_elastic(-0.5) is False
        assert is_elastic(0.8) is False

    def test_unit_elastic_is_not_elastic(self) -> None:
        assert is_elastic(-1.0) is False
        assert is_elastic(1.0) is False


class TestRevenueAtPrice:
    def test_zero_price_returns_zero(self) -> None:
        assert revenue_at_price(0.0, 100.0, 10.0, -1.5) == pytest.approx(0.0)

    def test_zero_base_price_returns_zero(self) -> None:
        assert revenue_at_price(10.0, 100.0, 0.0, -1.5) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        result = revenue_at_price(10.0, 100.0, 10.0, -1.5)
        assert isinstance(result, float)

    def test_positive_result_for_valid_inputs(self) -> None:
        result = revenue_at_price(10.0, 100.0, 10.0, -1.5)
        assert result > 0.0


class TestOptimalPriceDirection:
    def test_elastic_demand_suggests_lower(self) -> None:
        assert optimal_price_direction(10.0, -2.0) == "lower"

    def test_inelastic_demand_suggests_raise(self) -> None:
        assert optimal_price_direction(10.0, -0.5) == "raise"

    def test_unit_elastic_hold(self) -> None:
        assert optimal_price_direction(10.0, -1.0) == "hold"
        assert optimal_price_direction(10.0, 1.0) == "hold"
