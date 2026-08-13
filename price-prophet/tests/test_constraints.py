"""Tests for app/pricing/constraints.py."""

from __future__ import annotations


def test_apply_constraints_within_bounds():
    from app.pricing.constraints import PricingConstraints, apply_constraints

    c = PricingConstraints(min_price=1.0, max_price=500.0, max_change_pct=50.0)
    result = apply_constraints(120.0, 100.0, c)
    assert result == 120.0


def test_apply_constraints_clips_high():
    from app.pricing.constraints import PricingConstraints, apply_constraints

    c = PricingConstraints(max_change_pct=10.0)
    result = apply_constraints(200.0, 100.0, c)
    assert result <= 110.0


def test_apply_constraints_clips_low():
    from app.pricing.constraints import PricingConstraints, apply_constraints

    c = PricingConstraints(min_price=5.0, max_change_pct=10.0)
    result = apply_constraints(5.0, 100.0, c)
    assert result == 90.0


def test_violates_constraints_false():
    from app.pricing.constraints import PricingConstraints, violates_constraints

    c = PricingConstraints()
    assert violates_constraints(100.0, 100.0, c) is False


def test_violates_constraints_exceeds_max_change():
    from app.pricing.constraints import PricingConstraints, violates_constraints

    c = PricingConstraints(max_change_pct=10.0)
    assert violates_constraints(200.0, 100.0, c) is True


def test_violates_constraints_below_min_price():
    from app.pricing.constraints import PricingConstraints, violates_constraints

    c = PricingConstraints(min_price=50.0)
    assert violates_constraints(10.0, 100.0, c) is True
