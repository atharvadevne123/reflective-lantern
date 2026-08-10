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
    def test_purchase_before_signup_is_flagged(self):
        warnings = check_temporal_coherence(
            {"days_since_registration": 30, "days_since_last_purchase": 200}
        )
        assert len(warnings) == 1
        assert "predate signup" in warnings[0]

    def test_coherent_history_passes(self):
        warnings = check_temporal_coherence(
            {"days_since_registration": 365, "days_since_last_purchase": 30}
        )
        assert warnings == []

    def test_equal_values_pass(self):
        warnings = check_temporal_coherence(
            {"days_since_registration": 100, "days_since_last_purchase": 100}
        )
        assert warnings == []

    def test_decade_old_purchase_with_positive_count_flagged(self):
        warnings = check_temporal_coherence(
            {
                "days_since_registration": 5000,
                "days_since_last_purchase": 4000,
                "purchase_count": 5,
            }
        )
        assert any("decade" in w for w in warnings)

    def test_missing_fields_produce_no_warnings(self):
        assert check_temporal_coherence({}) == []


class TestEngagementCoherence:
    def test_clicks_exceeding_views_flagged(self):
        warnings = check_engagement_coherence({"view_count": 2, "click_count": 9})
        assert len(warnings) == 1
        assert "implies a view" in warnings[0]

    def test_clicks_below_views_pass(self):
        assert check_engagement_coherence({"view_count": 10, "click_count": 3}) == []

    def test_equal_clicks_and_views_pass(self):
        assert check_engagement_coherence({"view_count": 5, "click_count": 5}) == []

    def test_zero_engagement_passes(self):
        assert check_engagement_coherence({"view_count": 0, "click_count": 0}) == []

    def test_missing_fields_produce_no_warnings(self):
        assert check_engagement_coherence({}) == []


class TestCatalogCoherence:
    def test_rating_without_reviews_flagged(self):
        warnings = check_catalog_coherence({"item_avg_rating": 4.5, "item_review_count": 0})
        assert len(warnings) == 1

    def test_rating_with_reviews_passes(self):
        assert check_catalog_coherence({"item_avg_rating": 4.5, "item_review_count": 12}) == []

    def test_zero_rating_zero_reviews_passes(self):
        assert check_catalog_coherence({"item_avg_rating": 0.0, "item_review_count": 0}) == []


class TestValidatePayload:
    def test_collects_warnings_from_all_checks(self):
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

    def test_clean_payload_returns_empty(self):
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

    def test_empty_payload_returns_empty(self):
        assert validate_payload({}) == []

    @pytest.mark.parametrize(
        "reg,purchase,expected",
        [(365, 14, 0), (30, 200, 1), (100, 100, 0), (10, 11, 1)],
    )
    def test_temporal_boundary_cases(self, reg, purchase, expected):
        warnings = validate_payload(
            {"days_since_registration": reg, "days_since_last_purchase": purchase}
        )
        assert len(warnings) == expected


class TestValidationIssue:
    def test_carries_message_and_field(self):
        exc = ValidationIssue("bad value", field="user_age")
        assert exc.message == "bad value"
        assert exc.field == "user_age"

    def test_is_raisable(self):
        with pytest.raises(ValidationIssue):
            raise ValidationIssue("boom")


class TestValidationIntegration:
    def test_incoherent_payload_downgrades_confidence(self, client, intent_payload):
        """An incoherent payload must never come back as a high-confidence prediction."""
        payload = {**intent_payload, "view_count": 1, "click_count": 20}
        data = client.post("/api/v1/predict", json=payload).json()
        assert data["confidence"] == "low"
        assert len(data["warnings"]) >= 1

    def test_coherent_payload_has_no_warnings(self, client, intent_payload):
        data = client.post("/api/v1/predict", json=intent_payload).json()
        assert data["warnings"] == []
        assert data["confidence"] in ("high", "medium")


class TestValidateBatch:
    """Tests for the validate_batch helper added in the improvement run."""

    def test_empty_batch_returns_empty(self):
        from app.validation import validate_batch

        assert validate_batch([]) == {}

    def test_all_valid_returns_empty(self):
        from app.validation import validate_batch

        payloads = [
            {"days_since_registration": 100, "days_since_last_purchase": 50},
            {"view_count": 10, "click_count": 5},
        ]
        assert validate_batch(payloads) == {}

    def test_one_invalid_returns_index(self):
        from app.validation import validate_batch

        payloads = [
            {"days_since_registration": 10, "days_since_last_purchase": 200},
        ]
        result = validate_batch(payloads)
        assert "0" in result
        assert len(result["0"]) >= 1

    def test_mixed_batch_reports_only_invalid(self):
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
    def test_batch_parametrized(self, payloads, expected_invalid_count):
        from app.validation import validate_batch

        result = validate_batch(payloads)
        assert len(result) == expected_invalid_count


class TestCheckPriceCoherence:
    from app.validation import check_price_coherence

    def test_negative_discount_flagged(self):
        from app.validation import check_price_coherence

        warnings = check_price_coherence({"item_discount_pct": -5})
        assert any("negative" in w for w in warnings)

    def test_discount_over_100_flagged(self):
        from app.validation import check_price_coherence

        warnings = check_price_coherence({"item_discount_pct": 110})
        assert any("100" in w for w in warnings)

    def test_valid_discount_passes(self):
        from app.validation import check_price_coherence

        assert check_price_coherence({"item_discount_pct": 20}) == []

    def test_negative_price_flagged(self):
        from app.validation import check_price_coherence

        warnings = check_price_coherence({"item_price": -1.0})
        assert any("negative" in w for w in warnings)

    def test_negative_avg_order_flagged(self):
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
    def test_parametrized_price_checks(self, payload, expected_count):
        from app.validation import check_price_coherence

        assert len(check_price_coherence(payload)) == expected_count


class TestSanitisePayload:
    def test_negative_discount_clamped_to_zero(self):
        from app.validation import sanitise_payload

        result = sanitise_payload({"item_discount_pct": -20})
        assert result["item_discount_pct"] == 0.0

    def test_discount_over_100_clamped(self):
        from app.validation import sanitise_payload

        result = sanitise_payload({"item_discount_pct": 150})
        assert result["item_discount_pct"] == 100.0

    def test_negative_price_clamped_to_zero(self):
        from app.validation import sanitise_payload

        result = sanitise_payload({"item_price": -5.0})
        assert result["item_price"] == 0.0

    def test_negative_avg_order_clamped(self):
        from app.validation import sanitise_payload

        result = sanitise_payload({"avg_order_value": -100})
        assert result["avg_order_value"] == 0.0

    def test_cart_abandon_rate_clamped(self):
        from app.validation import sanitise_payload

        result = sanitise_payload({"cart_abandon_rate": 1.5})
        assert result["cart_abandon_rate"] == 1.0

    def test_valid_values_unchanged(self):
        from app.validation import sanitise_payload

        payload = {"item_discount_pct": 10, "item_price": 9.99, "avg_order_value": 50.0}
        result = sanitise_payload(payload)
        assert result["item_discount_pct"] == 10.0
        assert result["item_price"] == 9.99
        assert result["avg_order_value"] == 50.0

    def test_original_not_mutated(self):
        from app.validation import sanitise_payload

        payload = {"item_discount_pct": -50}
        sanitise_payload(payload)
        assert payload["item_discount_pct"] == -50


class TestValidationSummary:
    def test_empty_results_returns_zeros(self):
        from app.validation import validation_summary

        summary = validation_summary({})
        assert summary["total_warnings"] == 0
        assert summary["affected_payloads"] == 0
        assert summary["most_common_warning"] is None

    def test_single_payload_counted(self):
        from app.validation import validation_summary

        results = {"0": ["warning one", "warning two"]}
        summary = validation_summary(results)
        assert summary["total_warnings"] == 2
        assert summary["affected_payloads"] == 1

    def test_most_common_warning_identified(self):
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
    def test_parametrized_summary(self, results, total, affected):
        from app.validation import validation_summary

        summary = validation_summary(results)
        assert summary["total_warnings"] == total
        assert summary["affected_payloads"] == affected
