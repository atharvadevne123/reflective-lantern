"""Tests for compute_gross_yield."""

import pytest

from app.features import compute_gross_yield


def test_basic():
    assert compute_gross_yield(25_000.0, 500_000.0) == 5.0


def test_zero_price():
    assert compute_gross_yield(25_000.0, 0.0) == 0.0


def test_negative_price():
    assert compute_gross_yield(25_000.0, -100.0) == 0.0


@pytest.mark.parametrize("rent,price,expected", [
    (30_000, 600_000, 5.0),
    (12_000, 200_000, 6.0),
])
def test_parametrized(rent, price, expected):
    assert compute_gross_yield(rent, price) == expected
