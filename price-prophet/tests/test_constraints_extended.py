"""Extended pricing constraints tests with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.pricing.constraints import (
    PricingConstraints,
    apply_constraints,
    batch_apply_constraints,
    constraint_headroom,
    violates_constraints,
)


@pytest.mark.parametrize(
    "price,base,max_change_pct,expected",
    [
        (110.0, 100.0, 50.0, 110.0),
        (200.0, 100.0, 50.0, 150.0),
        (40.0, 100.0, 50.0, 50.0),
        (100.0, 100.0, 10.0, 100.0),
        (120.0, 100.0, 10.0, 110.0),
        (80.0, 100.0, 10.0, 90.0),
    ],
)
def test_apply_constraints_max_change_pct(
    price: float, base: float, max_change_pct: float, expected: float
) -> None:
    c = PricingConstraints(max_change_pct=max_change_pct)
    result = apply_constraints(price, base, c)
    assert result == pytest.approx(expected, rel=1e-6)


def test_apply_constraints_absolute_floor() -> None:
    c = PricingConstraints(min_price=10.0, max_change_pct=100.0)
    result = apply_constraints(5.0, 100.0, c)
    assert result >= 10.0


def test_apply_constraints_absolute_ceiling() -> None:
    c = PricingConstraints(max_price=200.0, max_change_pct=100.0)
    result = apply_constraints(500.0, 100.0, c)
    assert result <= 200.0


def test_apply_constraints_zero_base_no_change_guard() -> None:
    c = PricingConstraints(max_change_pct=10.0)
    result = apply_constraints(500.0, 0.0, c)
    assert result <= c.max_price


@pytest.mark.parametrize(
    "price,base,max_change_pct,expected_violation",
    [
        (110.0, 100.0, 50.0, False),
        (200.0, 100.0, 50.0, True),
        (50.0, 100.0, 50.0, False),
        (40.0, 100.0, 50.0, True),
        (100.0, 100.0, 0.0, False),
    ],
)
def test_violates_constraints_parametrized(
    price: float, base: float, max_change_pct: float, expected_violation: bool
) -> None:
    c = PricingConstraints(max_change_pct=max_change_pct)
    assert violates_constraints(price, base, c) is expected_violation


def test_violates_constraints_below_min_price() -> None:
    c = PricingConstraints(min_price=10.0)
    assert violates_constraints(5.0, 100.0, c) is True


def test_violates_constraints_above_max_price() -> None:
    c = PricingConstraints(max_price=200.0)
    assert violates_constraints(300.0, 100.0, c) is True


def test_constraint_headroom_has_required_keys() -> None:
    c = PricingConstraints(min_price=0.0, max_price=1000.0)
    result = constraint_headroom(100.0, c)
    assert "room_to_min" in result
    assert "room_to_max" in result


def test_constraint_headroom_at_min() -> None:
    c = PricingConstraints(min_price=100.0, max_price=200.0)
    result = constraint_headroom(100.0, c)
    assert result["room_to_min"] == pytest.approx(0.0)


def test_constraint_headroom_at_max() -> None:
    c = PricingConstraints(min_price=0.0, max_price=100.0)
    result = constraint_headroom(100.0, c)
    assert result["room_to_max"] == pytest.approx(0.0)


def test_constraint_headroom_non_negative() -> None:
    c = PricingConstraints(min_price=10.0, max_price=200.0)
    result = constraint_headroom(100.0, c)
    assert result["room_to_min"] >= 0.0
    assert result["room_to_max"] >= 0.0


def test_batch_apply_constraints_all_prices() -> None:
    c = PricingConstraints(max_change_pct=10.0)
    prices = [80.0, 90.0, 100.0, 110.0, 120.0]
    result = batch_apply_constraints(prices, c, base_price=100.0)
    assert len(result) == len(prices)
    assert all(90.0 <= p <= 110.0 for p in result)


def test_batch_apply_constraints_empty_list() -> None:
    c = PricingConstraints()
    assert batch_apply_constraints([], c) == []


@pytest.mark.parametrize("n", [1, 5, 10])
def test_batch_apply_constraints_length_preserved(n: int) -> None:
    c = PricingConstraints()
    prices = [100.0 * i for i in range(1, n + 1)]
    result = batch_apply_constraints(prices, c)
    assert len(result) == n
