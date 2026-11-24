"""Extended elasticity tests with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.pricing.elasticity import (
    apply_elasticity,
    estimate_elasticity,
    is_elastic,
    optimal_price_direction,
    revenue_at_price,
)


def test_estimate_elasticity_mismatched_lengths_returns_zero() -> None:
    assert estimate_elasticity([1.0, 2.0], [1.0]) == 0.0


def test_estimate_elasticity_fewer_than_two_valid_pairs_returns_zero() -> None:
    assert estimate_elasticity([1.0], [1.0]) == 0.0


def test_estimate_elasticity_nonpositive_prices_excluded() -> None:
    result = estimate_elasticity([0.0, -1.0], [100.0, 90.0])
    assert result == 0.0


def test_estimate_elasticity_constant_prices_zero_variance_returns_zero() -> None:
    result = estimate_elasticity([10.0, 10.0, 10.0], [100.0, 90.0, 80.0])
    assert result == 0.0


def test_estimate_elasticity_elastic_demand() -> None:
    prices = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    demands = [1000.0, 700.0, 450.0, 280.0, 180.0, 100.0]
    result = estimate_elasticity(prices, demands)
    assert result < -1.0


def test_apply_elasticity_zero_base_price_returns_base_demand() -> None:
    result = apply_elasticity(100.0, 0.0, 10.0, -1.5)
    assert result == 100.0


def test_apply_elasticity_zero_new_price_returns_base_demand() -> None:
    result = apply_elasticity(100.0, 10.0, 0.0, -1.5)
    assert result == 100.0


def test_apply_elasticity_same_price_returns_base_demand() -> None:
    result = apply_elasticity(100.0, 10.0, 10.0, -1.5)
    assert result == pytest.approx(100.0, rel=1e-6)


@pytest.mark.parametrize("elasticity", [-0.5, -1.0, -1.5, -2.0])
def test_apply_elasticity_price_increase_reduces_demand_for_negative_elasticity(
    elasticity: float,
) -> None:
    result = apply_elasticity(100.0, 10.0, 15.0, elasticity)
    assert result < 100.0


@pytest.mark.parametrize("elasticity", [-0.5, -1.0, -1.5, -2.0])
def test_apply_elasticity_price_decrease_increases_demand(elasticity: float) -> None:
    result = apply_elasticity(100.0, 10.0, 8.0, elasticity)
    assert result > 100.0


@pytest.mark.parametrize("elasticity,expected_elastic", [
    (-2.0, True),
    (-1.5, True),
    (-1.1, True),
    (-0.9, False),
    (-0.5, False),
    (0.5, False),
    (2.0, True),
])
def test_is_elastic_parametrized(elasticity: float, expected_elastic: bool) -> None:
    assert is_elastic(elasticity) is expected_elastic


@pytest.mark.parametrize("elasticity,expected_direction", [
    (-2.0, "lower"),
    (-1.5, "lower"),
    (-0.5, "raise"),
    (-1.0, "hold"),
    (1.0, "hold"),
])
def test_optimal_price_direction_parametrized(
    elasticity: float, expected_direction: str
) -> None:
    result = optimal_price_direction(100.0, elasticity)
    assert result == expected_direction


def test_revenue_at_price_zero_price_returns_zero() -> None:
    assert revenue_at_price(0.0, 100.0, 10.0, -1.5) == 0.0


def test_revenue_at_price_zero_base_price_returns_zero() -> None:
    assert revenue_at_price(10.0, 100.0, 0.0, -1.5) == 0.0


def test_revenue_at_price_non_negative() -> None:
    result = revenue_at_price(12.0, 100.0, 10.0, -1.5)
    assert result >= 0.0
