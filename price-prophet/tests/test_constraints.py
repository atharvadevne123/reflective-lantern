"""Tests for app/pricing/constraints.py."""

from __future__ import annotations

import pytest


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


@pytest.mark.parametrize(
    "new_price,base_price,max_change_pct,should_violate",
    [
        (105.0, 100.0, 10.0, False),
        (115.0, 100.0, 10.0, True),
        (90.0, 100.0, 15.0, False),
        (80.0, 100.0, 15.0, True),
    ],
)
def test_max_change_pct_parametrized(
    new_price: float, base_price: float, max_change_pct: float, should_violate: bool
) -> None:
    from app.pricing.constraints import PricingConstraints, violates_constraints

    c = PricingConstraints(max_change_pct=max_change_pct)
    assert violates_constraints(new_price, base_price, c) == should_violate


@pytest.mark.parametrize("min_price,max_price", [(0.0, 1000.0), (10.0, 500.0), (1.0, 9999.0)])
def test_apply_constraints_within_min_max(min_price: float, max_price: float) -> None:
    from app.pricing.constraints import PricingConstraints, apply_constraints

    c = PricingConstraints(min_price=min_price, max_price=max_price)
    result = apply_constraints(max_price / 2, max_price / 2, c)
    assert min_price <= result <= max_price


class TestConstraintHeadroom:
    def test_room_within_bounds(self) -> None:
        from app.pricing.constraints import PricingConstraints, constraint_headroom

        c = PricingConstraints(min_price=5.0, max_price=20.0, max_change_pct=50.0)
        result = constraint_headroom(10.0, c)
        assert result["room_to_min"] == pytest.approx(5.0)
        assert result["room_to_max"] == pytest.approx(10.0)

    def test_at_minimum_room_to_min_zero(self) -> None:
        from app.pricing.constraints import PricingConstraints, constraint_headroom

        c = PricingConstraints(min_price=5.0, max_price=20.0, max_change_pct=50.0)
        result = constraint_headroom(5.0, c)
        assert result["room_to_min"] == 0.0

    def test_at_maximum_room_to_max_zero(self) -> None:
        from app.pricing.constraints import PricingConstraints, constraint_headroom

        c = PricingConstraints(min_price=5.0, max_price=20.0, max_change_pct=50.0)
        result = constraint_headroom(20.0, c)
        assert result["room_to_max"] == 0.0

    def test_below_min_room_to_min_floored(self) -> None:
        from app.pricing.constraints import PricingConstraints, constraint_headroom

        c = PricingConstraints(min_price=10.0, max_price=50.0, max_change_pct=50.0)
        result = constraint_headroom(5.0, c)
        assert result["room_to_min"] == 0.0


class TestBatchApplyConstraints:
    def test_clips_low_prices(self) -> None:
        from app.pricing.constraints import PricingConstraints, batch_apply_constraints

        c = PricingConstraints(min_price=10.0, max_price=50.0, max_change_pct=100.0)
        result = batch_apply_constraints([1.0, 5.0, 15.0], c)
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(10.0)

    def test_clips_high_prices(self) -> None:
        from app.pricing.constraints import PricingConstraints, batch_apply_constraints

        c = PricingConstraints(min_price=1.0, max_price=20.0, max_change_pct=100.0)
        result = batch_apply_constraints([25.0, 30.0], c)
        assert all(p <= 20.0 for p in result)

    def test_same_length_output(self) -> None:
        from app.pricing.constraints import PricingConstraints, batch_apply_constraints

        c = PricingConstraints(min_price=1.0, max_price=100.0, max_change_pct=50.0)
        prices = [10.0, 20.0, 30.0, 40.0]
        assert len(batch_apply_constraints(prices, c)) == len(prices)
