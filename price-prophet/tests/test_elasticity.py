"""Tests for app/pricing/elasticity.py."""
from __future__ import annotations


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
