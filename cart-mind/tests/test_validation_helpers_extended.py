"""Extended tests for cart-mind app/validation.py helper functions."""

from __future__ import annotations

import pytest

from app.validation import (
    cart_value_tier,
    cross_field_penalty_score,
    item_count_flag,
    price_discount_pct,
    revenue_impact_score,
)


class TestPriceDiscountPct:
    def test_zero_when_no_discount(self) -> None:
        assert price_discount_pct(100.0, 100.0) == pytest.approx(0.0)

    def test_fifty_percent_discount(self) -> None:
        assert price_discount_pct(100.0, 50.0) == pytest.approx(50.0)

    def test_full_discount(self) -> None:
        assert price_discount_pct(100.0, 0.0) == pytest.approx(100.0)

    def test_small_discount(self) -> None:
        result = price_discount_pct(200.0, 190.0)
        assert result == pytest.approx(5.0, abs=0.1)


class TestCartValueTier:
    def test_low_tier(self) -> None:
        tier = cart_value_tier(10.0)
        assert isinstance(tier, str)
        assert len(tier) > 0

    def test_high_tier_differs(self) -> None:
        low = cart_value_tier(10.0)
        high = cart_value_tier(10_000.0)
        assert low != high

    def test_zero_value(self) -> None:
        tier = cart_value_tier(0.0)
        assert isinstance(tier, str)


class TestItemCountFlag:
    def test_below_threshold_false(self) -> None:
        assert item_count_flag(5) is False

    def test_at_threshold_true(self) -> None:
        assert item_count_flag(10) is True

    def test_above_threshold_true(self) -> None:
        assert item_count_flag(100) is True

    def test_custom_threshold(self) -> None:
        assert item_count_flag(5, bulk_threshold=5) is True
        assert item_count_flag(4, bulk_threshold=5) is False


class TestCrossFieldPenaltyScore:
    def test_no_warnings_zero_penalty(self) -> None:
        assert cross_field_penalty_score([]) == pytest.approx(0.0)

    def test_warnings_give_positive_penalty(self) -> None:
        score = cross_field_penalty_score(["warning1", "warning2"])
        assert score > 0.0

    def test_more_warnings_higher_penalty(self) -> None:
        few = cross_field_penalty_score(["w1"])
        many = cross_field_penalty_score(["w1", "w2", "w3"])
        assert many >= few


class TestRevenueImpactScore:
    def test_returns_float(self) -> None:
        result = revenue_impact_score(cart_value=100.0, confidence=0.9)
        assert isinstance(result, float)

    def test_higher_confidence_higher_score(self) -> None:
        low = revenue_impact_score(cart_value=100.0, confidence=0.1)
        high = revenue_impact_score(cart_value=100.0, confidence=0.9)
        assert high >= low

    def test_zero_cart_value(self) -> None:
        score = revenue_impact_score(cart_value=0.0, confidence=0.9)
        assert score == pytest.approx(0.0)
