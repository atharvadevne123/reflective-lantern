"""Tests for compute_price_per_sqft."""

import pytest

from app.features import compute_price_per_sqft


def test_basic():
    assert compute_price_per_sqft(300_000.0, 1500.0) == 200.0


def test_zero_sqft():
    assert compute_price_per_sqft(300_000.0, 0.0) == 0.0


def test_negative_sqft():
    assert compute_price_per_sqft(300_000.0, -100.0) == 0.0


@pytest.mark.parametrize("price,sqft,expected", [
    (500_000, 2000, 250.0),
    (150_000, 750, 200.0),
])
def test_parametrized(price, sqft, expected):
    assert compute_price_per_sqft(price, sqft) == expected
