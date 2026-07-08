"""Tests for market context enrichment functions."""

import math

import pytest

from app.market_context import (
    affordability_index,
    dom_classification,
    price_per_sqft,
    price_to_rent_ratio,
)


def test_price_per_sqft_basic():
    assert price_per_sqft(500_000, 1000) == pytest.approx(500.0)


def test_price_per_sqft_zero_sqft():
    assert price_per_sqft(500_000, 0) == 0.0


@pytest.mark.parametrize(
    "days,expected",
    [
        (5, "fast"),
        (13, "fast"),
        (14, "normal"),
        (60, "normal"),
        (61, "slow"),
        (365, "slow"),
    ],
)
def test_dom_classification(days, expected):
    assert dom_classification(days) == expected


def test_affordability_expensive_is_not_affordable():
    result = affordability_index(predicted_value=2_000_000, annual_income=80_000)
    assert result["is_affordable"] is False
    assert result["pct_income"] > 28


def test_affordability_cheap_is_affordable():
    result = affordability_index(predicted_value=150_000, annual_income=100_000)
    assert result["is_affordable"] is True


def test_affordability_contains_required_keys():
    result = affordability_index(500_000)
    assert "loan_amount" in result
    assert "monthly_payment" in result
    assert "pct_income" in result
    assert "is_affordable" in result


def test_affordability_loan_amount():
    result = affordability_index(500_000, down_payment_pct=0.20)
    assert result["loan_amount"] == pytest.approx(400_000.0)


def test_price_to_rent_ratio_basic():
    result = price_to_rent_ratio(500_000, 25_000)
    assert result == pytest.approx(20.0)


def test_price_to_rent_ratio_zero_rent():
    result = price_to_rent_ratio(500_000, 0)
    assert math.isinf(result)


@pytest.mark.parametrize(
    "value,rent,expect_buy",
    [
        (200_000, 20_000, True),
        (1_000_000, 20_000, False),
    ],
)
def test_buy_vs_rent_signal(value, rent, expect_buy):
    ratio = price_to_rent_ratio(value, rent)
    assert (ratio < 15) == expect_buy
