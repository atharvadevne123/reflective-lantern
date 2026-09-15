"""Tests for cross-field payload coherence validation."""

from __future__ import annotations

import pytest

from app.validation import (
    ValidationIssue,
    check_catalog_coherence,
    check_engagement_coherence,
    check_temporal_coherence,
    validate_payload,
)


class TestTemporalCoherence:
    def test_purchase_before_signup_is_flagged(self) -> None:
        warnings = check_temporal_coherence(
            {"days_since_registration": 30, "days_since_last_purchase": 200}
        )
        assert len(warnings) == 1
        assert "predate signup" in warnings[0]

    def test_coherent_history_passes(self) -> None:
        warnings = check_temporal_coherence(
            {"days_since_registration": 365, "days_since_last_purchase": 30}
        )
        assert warnings == []

    def test_equal_values_pass(self) -> None:
        warnings = check_temporal_coherence(
            {"days_since_registration": 100, "days_since_last_purchase": 100}
        )
        assert warnings == []

    def test_decade_old_purchase_with_positive_count_flagged(self) -> None:
        warnings = check_temporal_coherence(
            {
                "days_since_registration": 5000,
                "days_since_last_purchase": 4000,
                "purchase_count": 5,
            }
        )
        assert any("decade" in w for w in warnings)

    def test_missing_fields_produce_no_warnings(self) -> None:
        assert check_temporal_coherence({}) == []


class TestEngagementCoherence:
    def test_clicks_exceeding_views_flagged(self) -> None:
        warnings = check_engagement_coherence({"view_count": 2, "click_count": 9})
        assert len(warnings) == 1
        assert "implies a view" in warnings[0]

    def test_clicks_below_views_pass(self) -> None:
        assert check_engagement_coherence({"view_count": 10, "click_count": 3}) == []

    def test_equal_clicks_and_views_pass(self) -> None:
        assert check_engagement_coherence({"view_count": 5, "click_count": 5}) == []

    def test_zero_engagement_passes(self) -> None:
        assert check_engagement_coherence({"view_count": 0, "click_count": 0}) == []

    def test_missing_fields_produce_no_warnings(self) -> None:
        assert check_engagement_coherence({}) == []


class TestCatalogCoherence:
    def test_rating_without_reviews_flagged(self) -> None:
        warnings = check_catalog_coherence({"item_avg_rating": 4.5, "item_review_count": 0})
        assert len(warnings) == 1

    def test_rating_with_reviews_passes(self) -> None:
        assert check_catalog_coherence({"item_avg_rating": 4.5, "item_review_count": 12}) == []

    def test_zero_rating_zero_reviews_passes(self) -> None:
        assert check_catalog_coherence({"item_avg_rating": 0.0, "item_review_count": 0}) == []


class TestValidatePayload:
    def test_collects_warnings_from_all_checks(self) -> None:
        warnings = validate_payload(
            {
                "days_since_registration": 30,
                "days_since_last_purchase": 200,
                "view_count": 2,
                "click_count": 9,
                "item_avg_rating": 4.0,
                "item_review_count": 0,
            }
        )
        assert len(warnings) == 3

    def test_clean_payload_returns_empty(self) -> None:
        warnings = validate_payload(
            {
                "days_since_registration": 365,
                "days_since_last_purchase": 14,
                "view_count": 10,
                "click_count": 3,
                "item_avg_rating": 4.2,
                "item_review_count": 320,
            }
        )
        assert warnings == []

    def test_empty_payload_returns_empty(self) -> None:
        assert validate_payload({}) == []

    @pytest.mark.parametrize(
        "reg,purchase,expected",
        [(365, 14, 0), (30, 200, 1), (100, 100, 0), (10, 11, 1)],
    )
    def test_temporal_boundary_cases(self, reg, purchase, expected) -> None:
        warnings = validate_payload(
            {"days_since_registration": reg, "days_since_last_purchase": purchase}
        )
        assert len(warnings) == expected


class TestValidationIssue:
    def test_carries_message_and_field(self) -> None:
        exc = ValidationIssue("bad value", field="user_age")
        assert exc.message == "bad value"
        assert exc.field == "user_age"

    def test_is_raisable(self) -> None:
        with pytest.raises(ValidationIssue):
            raise ValidationIssue("boom")


class TestValidationIntegration:
    def test_incoherent_payload_downgrades_confidence(self, client, intent_payload) -> None:
        """An incoherent payload must never come back as a high-confidence prediction."""
        payload = {**intent_payload, "view_count": 1, "click_count": 20}
        data = client.post("/api/v1/predict", json=payload).json()
        assert data["confidence"] == "low"
        assert len(data["warnings"]) >= 1

    def test_coherent_payload_has_no_warnings(self, client, intent_payload) -> None:
        data = client.post("/api/v1/predict", json=intent_payload).json()
        assert data["warnings"] == []
        assert data["confidence"] in ("high", "medium")


class TestValidateBatch:
    """Tests for the validate_batch helper added in the improvement run."""

    def test_empty_batch_returns_empty(self) -> None:
        from app.validation import validate_batch

        assert validate_batch([]) == {}

    def test_all_valid_returns_empty(self) -> None:
        from app.validation import validate_batch

        payloads = [
            {"days_since_registration": 100, "days_since_last_purchase": 50},
            {"view_count": 10, "click_count": 5},
        ]
        assert validate_batch(payloads) == {}

    def test_one_invalid_returns_index(self) -> None:
        from app.validation import validate_batch

        payloads = [
            {"days_since_registration": 10, "days_since_last_purchase": 200},
        ]
        result = validate_batch(payloads)
        assert "0" in result
        assert len(result["0"]) >= 1

    def test_mixed_batch_reports_only_invalid(self) -> None:
        from app.validation import validate_batch

        payloads = [
            {"view_count": 5, "click_count": 1},  # valid
            {"view_count": 1, "click_count": 50},  # invalid
            {"days_since_registration": 30, "days_since_last_purchase": 10},  # valid
        ]
        result = validate_batch(payloads)
        assert set(result.keys()) == {"1"}

    @pytest.mark.parametrize(
        "payloads,expected_invalid_count",
        [
            ([], 0),
            ([{"view_count": 0, "click_count": 5}], 1),
            ([{"view_count": 5, "click_count": 1}] * 5, 0),
        ],
    )
    def test_batch_parametrized(self, payloads, expected_invalid_count) -> None:
        from app.validation import validate_batch

        result = validate_batch(payloads)
        assert len(result) == expected_invalid_count


class TestCheckPriceCoherence:
    from app.validation import check_price_coherence

    def test_negative_discount_flagged(self) -> None:
        from app.validation import check_price_coherence

        warnings = check_price_coherence({"item_discount_pct": -5})
        assert any("negative" in w for w in warnings)

    def test_discount_over_100_flagged(self) -> None:
        from app.validation import check_price_coherence

        warnings = check_price_coherence({"item_discount_pct": 110})
        assert any("100" in w for w in warnings)

    def test_valid_discount_passes(self) -> None:
        from app.validation import check_price_coherence

        assert check_price_coherence({"item_discount_pct": 20}) == []

    def test_negative_price_flagged(self) -> None:
        from app.validation import check_price_coherence

        warnings = check_price_coherence({"item_price": -1.0})
        assert any("negative" in w for w in warnings)

    def test_negative_avg_order_flagged(self) -> None:
        from app.validation import check_price_coherence

        warnings = check_price_coherence({"avg_order_value": -50})
        assert any("negative" in w for w in warnings)

    @pytest.mark.parametrize(
        "payload,expected_count",
        [
            ({"item_discount_pct": 50, "item_price": 9.99, "avg_order_value": 30.0}, 0),
            ({"item_discount_pct": -1, "item_price": -1.0}, 2),
            ({"item_discount_pct": 200, "avg_order_value": -10}, 2),
            ({}, 0),
        ],
    )
    def test_parametrized_price_checks(self, payload, expected_count) -> None:
        from app.validation import check_price_coherence

        assert len(check_price_coherence(payload)) == expected_count


class TestSanitisePayload:
    def test_negative_discount_clamped_to_zero(self) -> None:
        from app.validation import sanitise_payload

        result = sanitise_payload({"item_discount_pct": -20})
        assert result["item_discount_pct"] == 0.0

    def test_discount_over_100_clamped(self) -> None:
        from app.validation import sanitise_payload

        result = sanitise_payload({"item_discount_pct": 150})
        assert result["item_discount_pct"] == 100.0

    def test_negative_price_clamped_to_zero(self) -> None:
        from app.validation import sanitise_payload

        result = sanitise_payload({"item_price": -5.0})
        assert result["item_price"] == 0.0

    def test_negative_avg_order_clamped(self) -> None:
        from app.validation import sanitise_payload

        result = sanitise_payload({"avg_order_value": -100})
        assert result["avg_order_value"] == 0.0

    def test_cart_abandon_rate_clamped(self) -> None:
        from app.validation import sanitise_payload

        result = sanitise_payload({"cart_abandon_rate": 1.5})
        assert result["cart_abandon_rate"] == 1.0

    def test_valid_values_unchanged(self) -> None:
        from app.validation import sanitise_payload

        payload = {"item_discount_pct": 10, "item_price": 9.99, "avg_order_value": 50.0}
        result = sanitise_payload(payload)
        assert result["item_discount_pct"] == 10.0
        assert result["item_price"] == 9.99
        assert result["avg_order_value"] == 50.0

    def test_original_not_mutated(self) -> None:
        from app.validation import sanitise_payload

        payload = {"item_discount_pct": -50}
        sanitise_payload(payload)
        assert payload["item_discount_pct"] == -50


class TestValidationSummary:
    def test_empty_results_returns_zeros(self) -> None:
        from app.validation import validation_summary

        summary = validation_summary({})
        assert summary["total_warnings"] == 0
        assert summary["affected_payloads"] == 0
        assert summary["most_common_warning"] is None

    def test_single_payload_counted(self) -> None:
        from app.validation import validation_summary

        results = {"0": ["warning one", "warning two"]}
        summary = validation_summary(results)
        assert summary["total_warnings"] == 2
        assert summary["affected_payloads"] == 1

    def test_most_common_warning_identified(self) -> None:
        from app.validation import validation_summary

        results = {
            "0": ["warn A", "warn B"],
            "1": ["warn A"],
            "2": ["warn B", "warn B"],
        }
        summary = validation_summary(results)
        assert summary["most_common_warning"] == "warn B"

    @pytest.mark.parametrize(
        "results,total,affected",
        [
            ({}, 0, 0),
            ({"0": ["x"]}, 1, 1),
            ({"0": ["x", "y"], "1": ["x"]}, 3, 2),
        ],
    )
    def test_parametrized_summary(self, results, total, affected) -> None:
        from app.validation import validation_summary

        summary = validation_summary(results)
        assert summary["total_warnings"] == total
        assert summary["affected_payloads"] == affected


class TestPriceDiscountPct:
    def test_50_percent_off(self) -> None:
        from app.validation import price_discount_pct

        assert price_discount_pct(100.0, 50.0) == pytest.approx(50.0, abs=0.01)

    def test_no_discount(self) -> None:
        from app.validation import price_discount_pct

        assert price_discount_pct(100.0, 100.0) == pytest.approx(0.0, abs=0.01)

    def test_zero_original_returns_zero(self) -> None:
        from app.validation import price_discount_pct

        assert price_discount_pct(0.0, 50.0) == 0.0

    def test_clamped_to_100(self) -> None:
        from app.validation import price_discount_pct

        assert price_discount_pct(10.0, -5.0) == pytest.approx(100.0, abs=0.01)


class TestCartValueTier:
    def test_low(self) -> None:
        from app.validation import cart_value_tier

        assert cart_value_tier(10.0) == "low"

    def test_medium(self) -> None:
        from app.validation import cart_value_tier

        assert cart_value_tier(50.0) == "medium"

    def test_high(self) -> None:
        from app.validation import cart_value_tier

        assert cart_value_tier(200.0) == "high"

    def test_premium(self) -> None:
        from app.validation import cart_value_tier

        assert cart_value_tier(1000.0) == "premium"


class TestItemCountFlag:
    def test_below_threshold(self) -> None:
        from app.validation import item_count_flag

        assert item_count_flag(5) is False

    def test_at_threshold(self) -> None:
        from app.validation import item_count_flag

        assert item_count_flag(10) is True

    def test_above_threshold(self) -> None:
        from app.validation import item_count_flag

        assert item_count_flag(20) is True

    def test_custom_threshold(self) -> None:
        from app.validation import item_count_flag

        assert item_count_flag(3, bulk_threshold=5) is False
        assert item_count_flag(5, bulk_threshold=5) is True


class TestCrossFieldPenaltyScore:
    def test_no_warnings_zero_penalty(self) -> None:
        from app.validation import cross_field_penalty_score

        assert cross_field_penalty_score([]) == 0.0

    def test_one_warning_is_01(self) -> None:
        from app.validation import cross_field_penalty_score

        assert cross_field_penalty_score(["w1"]) == pytest.approx(0.1)

    def test_capped_at_one(self) -> None:
        from app.validation import cross_field_penalty_score

        result = cross_field_penalty_score(["w"] * 20)
        assert result == 1.0

    @pytest.mark.parametrize("n,expected", [(0, 0.0), (1, 0.1), (5, 0.5)])
    def test_parametrized_counts(self, n: int, expected: float) -> None:
        from app.validation import cross_field_penalty_score

        assert cross_field_penalty_score(["x"] * n) == pytest.approx(expected)


class TestNormalisePayload:
    def test_fills_missing_key(self) -> None:
        from app.validation import normalise_payload

        result = normalise_payload({}, defaults={"item_price": 10.0})
        assert result["item_price"] == 10.0

    def test_payload_overrides_default(self) -> None:
        from app.validation import normalise_payload

        result = normalise_payload({"item_price": 99.0}, defaults={"item_price": 10.0})
        assert result["item_price"] == 99.0

    def test_none_defaults_returns_copy(self) -> None:
        from app.validation import normalise_payload

        payload = {"a": 1}
        result = normalise_payload(payload)
        assert result == {"a": 1}
        assert result is not payload

    def test_extra_payload_keys_preserved(self) -> None:
        from app.validation import normalise_payload

        result = normalise_payload({"extra": 42}, defaults={"base": 0})
        assert result["extra"] == 42
        assert result["base"] == 0


class TestCheckQuantityCoherence:
    def test_cart_value_with_zero_items_flagged(self) -> None:
        from app.validation import check_quantity_coherence

        warnings = check_quantity_coherence({"cart_item_count": 0, "cart_value": 50.0})
        assert len(warnings) == 1
        assert "cart_item_count is zero" in warnings[0]

    def test_positive_items_with_value_passes(self) -> None:
        from app.validation import check_quantity_coherence

        assert check_quantity_coherence({"cart_item_count": 3, "cart_value": 50.0}) == []

    def test_zero_items_zero_value_passes(self) -> None:
        from app.validation import check_quantity_coherence

        assert check_quantity_coherence({"cart_item_count": 0, "cart_value": 0.0}) == []

    def test_avg_order_with_zero_purchases_flagged(self) -> None:
        from app.validation import check_quantity_coherence

        warnings = check_quantity_coherence({"purchase_count": 0, "avg_order_value": 99.0})
        assert len(warnings) == 1
        assert "purchase_count is zero" in warnings[0]

    def test_positive_purchases_with_avg_order_passes(self) -> None:
        from app.validation import check_quantity_coherence

        assert check_quantity_coherence({"purchase_count": 5, "avg_order_value": 30.0}) == []

    def test_missing_fields_no_warnings(self) -> None:
        from app.validation import check_quantity_coherence

        assert check_quantity_coherence({}) == []

    @pytest.mark.parametrize(
        "payload,expected_count",
        [
            ({"cart_item_count": 0, "cart_value": 10.0, "purchase_count": 0, "avg_order_value": 50.0}, 2),
            ({"cart_item_count": 1, "cart_value": 10.0, "purchase_count": 1, "avg_order_value": 10.0}, 0),
            ({}, 0),
        ],
    )
    def test_parametrized_quantity_checks(self, payload: dict, expected_count: int) -> None:
        from app.validation import check_quantity_coherence

        assert len(check_quantity_coherence(payload)) == expected_count


class TestTemporalCoherenceEdgeCases:
    def test_zero_days_registration_and_purchase_passes(self) -> None:
        assert check_temporal_coherence(
            {"days_since_registration": 0, "days_since_last_purchase": 0}
        ) == []

    def test_exactly_one_day_difference_passes(self) -> None:
        assert check_temporal_coherence(
            {"days_since_registration": 10, "days_since_last_purchase": 9}
        ) == []

    def test_purchase_count_zero_with_old_purchase_no_warning(self) -> None:
        warnings = check_temporal_coherence(
            {"days_since_registration": 5000, "days_since_last_purchase": 4000, "purchase_count": 0}
        )
        assert not any("decade" in w for w in warnings)

    @pytest.mark.parametrize("count", [1, 10, 100])
    def test_decade_warning_fires_for_any_positive_count(self, count: int) -> None:
        warnings = check_temporal_coherence(
            {"days_since_registration": 5000, "days_since_last_purchase": 4000, "purchase_count": count}
        )
        assert any("decade" in w for w in warnings)


class TestEngagementCoherenceExtended:
    @pytest.mark.parametrize("views,clicks", [(0, 1), (3, 10), (1, 2)])
    def test_clicks_exceed_views_parametrized(self, views: int, clicks: int) -> None:
        from app.validation import check_engagement_coherence

        warnings = check_engagement_coherence({"view_count": views, "click_count": clicks})
        assert len(warnings) == 1

    @pytest.mark.parametrize("views,clicks", [(10, 10), (5, 0), (100, 99)])
    def test_valid_engagement_parametrized(self, views: int, clicks: int) -> None:
        from app.validation import check_engagement_coherence

        assert check_engagement_coherence({"view_count": views, "click_count": clicks}) == []

    def test_only_views_field_no_warning(self) -> None:
        from app.validation import check_engagement_coherence

        assert check_engagement_coherence({"view_count": 5}) == []

    def test_only_clicks_field_no_warning(self) -> None:
        from app.validation import check_engagement_coherence

        assert check_engagement_coherence({"click_count": 5}) == []


class TestCatalogCoherenceExtended:
    @pytest.mark.parametrize("rating,reviews", [(1.0, 0), (5.0, 0), (3.5, 0)])
    def test_positive_rating_no_reviews_flagged(self, rating: float, reviews: int) -> None:
        from app.validation import check_catalog_coherence

        warnings = check_catalog_coherence({"item_avg_rating": rating, "item_review_count": reviews})
        assert len(warnings) == 1

    def test_only_rating_no_warning(self) -> None:
        from app.validation import check_catalog_coherence

        assert check_catalog_coherence({"item_avg_rating": 4.5}) == []

    def test_only_review_count_no_warning(self) -> None:
        from app.validation import check_catalog_coherence

        assert check_catalog_coherence({"item_review_count": 10}) == []

    def test_high_reviews_no_warning(self) -> None:
        from app.validation import check_catalog_coherence

        assert check_catalog_coherence({"item_avg_rating": 4.9, "item_review_count": 1000}) == []


class TestCartValueTierExtended:
    @pytest.mark.parametrize("value,tier", [
        (0.0, "low"),
        (24.99, "low"),
        (25.0, "medium"),
        (99.99, "medium"),
        (100.0, "high"),
        (499.99, "high"),
        (500.0, "premium"),
        (9999.0, "premium"),
    ])
    def test_boundary_values(self, value: float, tier: str) -> None:
        from app.validation import cart_value_tier

        assert cart_value_tier(value) == tier


class TestPriceDiscountPctExtended:
    @pytest.mark.parametrize("original,discounted,expected", [
        (200.0, 150.0, 25.0),
        (10.0, 10.0, 0.0),
        (100.0, 0.0, 100.0),
        (50.0, 25.0, 50.0),
    ])
    def test_known_values(self, original: float, discounted: float, expected: float) -> None:
        from app.validation import price_discount_pct

        assert price_discount_pct(original, discounted) == pytest.approx(expected, abs=0.01)

    def test_result_always_between_0_and_100(self) -> None:
        from app.validation import price_discount_pct

        for orig, disc in [(100, 50), (10, 200), (0, 5), (5, 0)]:
            result = price_discount_pct(float(orig), float(disc))
            assert 0.0 <= result <= 100.0


class TestItemCountFlagExtended:
    @pytest.mark.parametrize("count,threshold,expected", [
        (0, 10, False),
        (9, 10, False),
        (10, 10, True),
        (100, 10, True),
        (1, 1, True),
        (0, 1, False),
    ])
    def test_boundary_values(self, count: int, threshold: int, expected: bool) -> None:
        from app.validation import item_count_flag

        assert item_count_flag(count, bulk_threshold=threshold) == expected


class TestRevenueImpactScore:
    def test_zero_cart_value_returns_zero(self) -> None:
        from app.validation import revenue_impact_score

        assert revenue_impact_score(0.0, 0.8) == 0.0

    def test_zero_probability_returns_zero(self) -> None:
        from app.validation import revenue_impact_score

        assert revenue_impact_score(100.0, 0.0) == 0.0

    def test_no_discount_full_probability(self) -> None:
        from app.validation import revenue_impact_score

        result = revenue_impact_score(100.0, 1.0, discount_pct=0.0)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_50_percent_discount(self) -> None:
        from app.validation import revenue_impact_score

        result = revenue_impact_score(100.0, 1.0, discount_pct=50.0)
        assert result == pytest.approx(50.0, abs=0.01)

    def test_negative_cart_value_returns_zero(self) -> None:
        from app.validation import revenue_impact_score

        assert revenue_impact_score(-10.0, 0.5) == 0.0

    def test_invalid_probability_returns_zero(self) -> None:
        from app.validation import revenue_impact_score

        assert revenue_impact_score(100.0, 1.5) == 0.0

    @pytest.mark.parametrize("cart,prob,discount,expected", [
        (200.0, 0.5, 0.0, 100.0),
        (100.0, 0.5, 50.0, 25.0),
        (50.0, 1.0, 100.0, 0.0),
    ])
    def test_parametrized_cases(self, cart, prob, discount, expected) -> None:
        from app.validation import revenue_impact_score

        assert revenue_impact_score(cart, prob, discount_pct=discount) == pytest.approx(expected, abs=0.01)


class TestCrossFieldPenaltyScoreExtended:
    def test_ten_warnings_is_one(self) -> None:
        from app.validation import cross_field_penalty_score

        assert cross_field_penalty_score(["w"] * 10) == pytest.approx(1.0)

    def test_two_warnings_is_02(self) -> None:
        from app.validation import cross_field_penalty_score

        assert cross_field_penalty_score(["a", "b"]) == pytest.approx(0.2)

    def test_penalty_never_exceeds_one(self) -> None:
        from app.validation import cross_field_penalty_score

        for n in range(0, 25):
            assert cross_field_penalty_score(["x"] * n) <= 1.0


class TestValidatePayloadWithQuantityCoherence:
    def test_quantity_incoherence_included_in_warnings(self) -> None:
        from app.validation import validate_payload

        payload = {"cart_item_count": 0, "cart_value": 150.0}
        warnings = validate_payload(payload)
        assert any("cart_item_count is zero" in w for w in warnings)

    def test_multiple_coherence_failures_all_reported(self) -> None:
        from app.validation import validate_payload

        payload = {
            "days_since_registration": 5,
            "days_since_last_purchase": 100,
            "view_count": 1,
            "click_count": 10,
            "cart_item_count": 0,
            "cart_value": 99.0,
        }
        warnings = validate_payload(payload)
        assert len(warnings) >= 3

    def test_fully_coherent_payload_no_warnings(self) -> None:
        from app.validation import validate_payload

        payload = {
            "days_since_registration": 365,
            "days_since_last_purchase": 30,
            "view_count": 10,
            "click_count": 3,
            "item_avg_rating": 4.2,
            "item_review_count": 200,
            "item_discount_pct": 15.0,
            "item_price": 49.99,
            "avg_order_value": 75.0,
            "cart_item_count": 2,
            "cart_value": 99.98,
            "purchase_count": 5,
        }
        warnings = validate_payload(payload)
        assert warnings == []
