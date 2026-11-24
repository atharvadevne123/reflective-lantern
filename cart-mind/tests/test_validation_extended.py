"""Extended validation tests with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.validation import (
    check_engagement_coherence,
    check_quantity_coherence,
    check_temporal_coherence,
    cross_field_penalty_score,
    normalise_payload,
)


def test_check_temporal_coherence_valid_no_warnings() -> None:
    payload = {"days_since_registration": 100, "days_since_last_purchase": 50}
    assert check_temporal_coherence(payload) == []


def test_check_temporal_coherence_purchase_before_registration() -> None:
    payload = {"days_since_registration": 30, "days_since_last_purchase": 50}
    warnings = check_temporal_coherence(payload)
    assert len(warnings) > 0


def test_check_temporal_coherence_missing_fields_no_warnings() -> None:
    assert check_temporal_coherence({}) == []


def test_check_temporal_coherence_equal_days_no_warning() -> None:
    payload = {"days_since_registration": 50, "days_since_last_purchase": 50}
    assert check_temporal_coherence(payload) == []


@pytest.mark.parametrize("registration,purchase,should_warn", [
    (100, 50, False),
    (50, 100, True),
    (0, 1, True),
    (100, 100, False),
])
def test_check_temporal_coherence_parametrized(
    registration: int, purchase: int, should_warn: bool
) -> None:
    payload = {"days_since_registration": registration, "days_since_last_purchase": purchase}
    warnings = check_temporal_coherence(payload)
    assert bool(warnings) == should_warn


def test_check_engagement_coherence_valid_clicks_with_views() -> None:
    payload = {"view_count": 100, "click_count": 10}
    assert check_engagement_coherence(payload) == []


def test_check_engagement_coherence_clicks_without_views_warns() -> None:
    payload = {"view_count": 0, "click_count": 10}
    warnings = check_engagement_coherence(payload)
    assert len(warnings) > 0


def test_check_engagement_coherence_missing_fields_no_warnings() -> None:
    assert check_engagement_coherence({}) == []


def test_check_quantity_coherence_zero_cart_zero_value_ok() -> None:
    payload = {"cart_item_count": 0, "cart_value": 0.0}
    assert check_quantity_coherence(payload) == []


def test_check_quantity_coherence_zero_items_positive_value_warns() -> None:
    payload = {"cart_item_count": 0, "cart_value": 50.0}
    warnings = check_quantity_coherence(payload)
    assert len(warnings) > 0


def test_check_quantity_coherence_positive_items_positive_value_ok() -> None:
    payload = {"cart_item_count": 3, "cart_value": 90.0}
    assert check_quantity_coherence(payload) == []


def test_check_quantity_coherence_zero_purchases_positive_avg_warns() -> None:
    payload = {"purchase_count": 0, "avg_order_value": 50.0}
    warnings = check_quantity_coherence(payload)
    assert len(warnings) > 0


def test_check_quantity_coherence_positive_purchases_positive_avg_ok() -> None:
    payload = {"purchase_count": 5, "avg_order_value": 50.0}
    assert check_quantity_coherence(payload) == []


@pytest.mark.parametrize("n_warnings,expected_penalty", [
    (0, 0.0),
    (1, 0.1),
    (5, 0.5),
    (10, 1.0),
    (15, 1.0),
])
def test_cross_field_penalty_score_parametrized(n_warnings: int, expected_penalty: float) -> None:
    warnings = ["warning"] * n_warnings
    result = cross_field_penalty_score(warnings)
    assert result == pytest.approx(expected_penalty, abs=1e-6)


def test_cross_field_penalty_score_empty_no_penalty() -> None:
    assert cross_field_penalty_score([]) == pytest.approx(0.0)


def test_cross_field_penalty_score_capped_at_one() -> None:
    warnings = ["w"] * 100
    result = cross_field_penalty_score(warnings)
    assert result <= 1.0


def test_normalise_payload_fills_missing_defaults() -> None:
    payload = {"a": 1}
    defaults = {"a": 99, "b": 2, "c": 3}
    result = normalise_payload(payload, defaults)
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_normalise_payload_no_defaults_returns_copy() -> None:
    payload = {"x": 10}
    result = normalise_payload(payload)
    assert result == payload
    assert result is not payload


def test_normalise_payload_original_not_mutated() -> None:
    payload = {"a": 1}
    defaults = {"a": 99, "b": 2}
    _ = normalise_payload(payload, defaults)
    assert payload == {"a": 1}
