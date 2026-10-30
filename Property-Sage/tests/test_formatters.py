"""Tests for app.formatters helpers."""

import pytest

from app.formatters import format_metrics_response, format_prediction_response


SAMPLE_PREDICTION = {
    "predicted_price": 450000.0,
    "predicted_rental_yield": 0.05,
    "estimated_annual_rental": 22500.0,
    "estimated_monthly_rental": 1875.0,
}


def test_format_prediction_keys():
    result = format_prediction_response(SAMPLE_PREDICTION, "req-123", "suburb")
    assert "request_id" in result
    assert "predicted_price_formatted" in result
    assert "predicted_rental_yield_pct" in result


def test_format_prediction_request_id():
    result = format_prediction_response(SAMPLE_PREDICTION, "abc-123", "downtown")
    assert result["request_id"] == "abc-123"


def test_format_prediction_yield_pct():
    result = format_prediction_response(SAMPLE_PREDICTION, "r1", "suburb")
    assert "5.00%" in result["predicted_rental_yield_pct"]


def test_format_metrics_empty():
    result = format_metrics_response({})
    assert result["status"] == "no_metrics_available"


def test_format_metrics_structure():
    metrics = {
        "price_r2_mean": 0.85, "price_r2_std": 0.02,
        "price_rmse": 25000.0, "price_mae": 15000.0,
        "rental_r2_mean": 0.72, "rental_r2_std": 0.03,
        "rental_rmse": 0.01, "rental_mae": 0.008,
        "n_train": 1000,
    }
    result = format_metrics_response(metrics)
    assert "price_model" in result
    assert "rental_model" in result
    assert result["n_train"] == 1000


@pytest.mark.parametrize("neighborhood", ["suburb", "downtown", "waterfront"])
def test_format_prediction_neighborhood(neighborhood):
    result = format_prediction_response(SAMPLE_PREDICTION, "r", neighborhood)
    assert result["neighborhood"] == neighborhood
