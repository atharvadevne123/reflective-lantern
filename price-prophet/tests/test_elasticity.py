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


@pytest.mark.parametrize("elasticity,expected", [(-2.0, True), (-1.0, False), (-0.5, False), (-1.001, True)])
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
