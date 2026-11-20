"""Extended tests for cart-mind app/validation.py."""

from __future__ import annotations

import pytest

from app.validation import (
    ValidationIssue,
    cart_value_tier,
    check_catalog_coherence,
    check_engagement_coherence,
    check_quantity_coherence,
    check_temporal_coherence,
    cross_field_penalty_score,
    item_count_flag,
    revenue_impact_score,
    sanitise_payload,
    validate_batch,
    validate_payload,
)


class TestValidationIssue:
    def test_message_attr(self) -> None:
        exc = ValidationIssue("bad input", field="x")
        assert exc.message == "bad input"

    def test_field_attr(self) -> None:
        exc = ValidationIssue("bad input", field="myfield")
        assert exc.field == "myfield"

    def test_field_none_by_default(self) -> None:
        exc = ValidationIssue("oops")
        assert exc.field is None


class TestCheckTemporalCoherence:
    def test_purchase_after_registration_ok(self) -> None:
        result = check_temporal_coherence({"days_since_registration": 30, "days_since_last_purchase": 10})
        assert result == []

    def test_purchase_before_registration_warns(self) -> None:
        result = check_temporal_coherence({"days_since_registration": 10, "days_since_last_purchase": 30})
        assert len(result) == 1

    def test_very_old_purchase_with_nonzero_count_warns(self) -> None:
        result = check_temporal_coherence({"purchase_count": 5, "days_since_last_purchase": 4000})
        assert len(result) == 1

    def test_missing_fields_ok(self) -> None:
        assert check_temporal_coherence({}) == []


class TestCheckEngagementCoherence:
    def test_more_clicks_than_views_warns(self) -> None:
        result = check_engagement_coherence({"view_count": 5, "click_count": 10})
        assert len(result) == 1

    def test_equal_clicks_views_ok(self) -> None:
        result = check_engagement_coherence({"view_count": 5, "click_count": 5})
        assert result == []

    def test_missing_fields_ok(self) -> None:
        assert check_engagement_coherence({}) == []


class TestCheckCatalogCoherence:
    def test_rating_with_zero_reviews_warns(self) -> None:
        result = check_catalog_coherence({"item_avg_rating": 4.5, "item_review_count": 0})
        assert len(result) == 1

    def test_rating_with_reviews_ok(self) -> None:
        result = check_catalog_coherence({"item_avg_rating": 4.5, "item_review_count": 10})
        assert result == []


class TestCheckQuantityCoherence:
    def test_positive_cart_value_zero_items_warns(self) -> None:
        result = check_quantity_coherence({"cart_item_count": 0, "cart_value": 50.0})
        assert len(result) == 1

    def test_positive_avg_order_zero_purchases_warns(self) -> None:
        result = check_quantity_coherence({"purchase_count": 0, "avg_order_value": 100.0})
        assert len(result) == 1

    def test_ok_when_consistent(self) -> None:
        result = check_quantity_coherence({"cart_item_count": 2, "cart_value": 30.0})
        assert result == []


class TestValidatePayload:
    def test_clean_payload_no_warnings(self) -> None:
        result = validate_payload({"days_since_registration": 10, "days_since_last_purchase": 5})
        assert result == []

    def test_dirty_payload_returns_warnings(self) -> None:
        result = validate_payload({"view_count": 0, "click_count": 5})
        assert len(result) > 0


class TestValidateBatch:
    def test_clean_batch_empty_result(self) -> None:
        payloads = [{"days_since_registration": 10}]
        result = validate_batch(payloads)
        assert result == {}

    def test_dirty_batch_includes_index(self) -> None:
        payloads = [{"view_count": 0, "click_count": 5}]
        result = validate_batch(payloads)
        assert "0" in result


class TestSanitisePayload:
    def test_discount_clamped_below_zero(self) -> None:
        result = sanitise_payload({"item_discount_pct": -10.0})
        assert result["item_discount_pct"] == pytest.approx(0.0)

    def test_discount_clamped_above_100(self) -> None:
        result = sanitise_payload({"item_discount_pct": 150.0})
        assert result["item_discount_pct"] == pytest.approx(100.0)

    def test_negative_price_clamped(self) -> None:
        result = sanitise_payload({"item_price": -5.0})
        assert result["item_price"] == pytest.approx(0.0)


class TestRevenueImpactScore:
    def test_zero_probability(self) -> None:
        assert revenue_impact_score(100.0, 0.0) == pytest.approx(0.0)

    def test_full_probability_no_discount(self) -> None:
        assert revenue_impact_score(100.0, 1.0, 0.0) == pytest.approx(100.0)

    def test_discount_reduces_revenue(self) -> None:
        result = revenue_impact_score(100.0, 1.0, 50.0)
        assert result == pytest.approx(50.0)

    def test_negative_cart_value_returns_zero(self) -> None:
        assert revenue_impact_score(-10.0, 0.5) == pytest.approx(0.0)


class TestCartValueTier:
    def test_low_tier(self) -> None:
        assert cart_value_tier(20.0) == "low"

    def test_medium_tier(self) -> None:
        assert cart_value_tier(50.0) == "medium"

    def test_high_tier(self) -> None:
        assert cart_value_tier(200.0) == "high"

    def test_premium_tier(self) -> None:
        assert cart_value_tier(600.0) == "premium"


class TestItemCountFlag:
    def test_below_threshold_false(self) -> None:
        assert item_count_flag(5) is False

    def test_at_threshold_true(self) -> None:
        assert item_count_flag(10) is True

    def test_above_threshold_true(self) -> None:
        assert item_count_flag(20) is True


class TestCrossFieldPenaltyScore:
    def test_no_warnings_zero_penalty(self) -> None:
        assert cross_field_penalty_score([]) == pytest.approx(0.0)

    def test_one_warning_0_1_penalty(self) -> None:
        assert cross_field_penalty_score(["w1"]) == pytest.approx(0.1)

    def test_ten_warnings_max_1_0(self) -> None:
        assert cross_field_penalty_score(["w"] * 10) == pytest.approx(1.0)
