"""Tests for compute_cap_rate helper."""

import pytest

from app.features import compute_cap_rate


def test_cap_rate_basic():
    rate = compute_cap_rate(50_000.0, 1_000_000.0)
    assert abs(rate - 0.05) < 1e-6


def test_cap_rate_zero_value():
    assert compute_cap_rate(50_000.0, 0.0) == 0.0


def test_cap_rate_negative_value():
    assert compute_cap_rate(50_000.0, -1.0) == 0.0


@pytest.mark.parametrize("income,value,expected", [
    (60_000, 1_200_000, 0.05),
    (30_000, 500_000, 0.06),
])
def test_cap_rate_parametrized(income, value, expected):
    assert abs(compute_cap_rate(income, value) - expected) < 1e-4
