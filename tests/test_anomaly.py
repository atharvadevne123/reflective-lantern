"""Tests for valuation anomaly detection."""

import pytest

from app.anomaly import detect_valuation_anomaly


def test_normal_value_not_anomaly():
    result = detect_valuation_anomaly(predicted=500_000, neighborhood_median=480_000)
    assert result["is_anomaly"] is False
    assert result["direction"] == "high"


def test_severely_underpriced_is_anomaly():
    result = detect_valuation_anomaly(predicted=100_000, neighborhood_median=800_000)
    assert result["is_anomaly"] is True
    assert result["direction"] == "low"


def test_severely_overpriced_is_anomaly():
    result = detect_valuation_anomaly(predicted=3_000_000, neighborhood_median=400_000)
    assert result["is_anomaly"] is True
    assert result["direction"] == "high"


def test_zscore_method_used_when_std_provided():
    result = detect_valuation_anomaly(
        predicted=2_000_000,
        neighborhood_median=500_000,
        neighborhood_std=50_000,
    )
    assert result["method"] == "zscore"
    assert result["is_anomaly"] is True


def test_zscore_within_threshold_not_anomaly():
    result = detect_valuation_anomaly(
        predicted=510_000,
        neighborhood_median=500_000,
        neighborhood_std=50_000,
    )
    assert result["method"] == "zscore"
    assert result["is_anomaly"] is False


def test_iqr_method_used_with_reference_values():
    ref = list(range(300_000, 600_000, 10_000))
    result = detect_valuation_anomaly(
        predicted=5_000_000,
        neighborhood_median=450_000,
        reference_values=ref,
    )
    assert result["method"] == "iqr"
    assert result["is_anomaly"] is True


def test_zero_predicted_skipped():
    result = detect_valuation_anomaly(predicted=0, neighborhood_median=500_000)
    assert result["is_anomaly"] is False
    assert result["method"] == "skipped"


@pytest.mark.parametrize(
    "predicted,median,expect_anomaly",
    [
        (500_000, 490_000, False),
        (200_000, 500_000, True),
        (1_500_000, 500_000, True),
        (550_000, 500_000, False),
    ],
)
def test_ratio_method_parametrized(predicted, median, expect_anomaly):
    result = detect_valuation_anomaly(predicted=predicted, neighborhood_median=median)
    assert result["is_anomaly"] == expect_anomaly


def test_deviation_pct_computed():
    result = detect_valuation_anomaly(predicted=600_000, neighborhood_median=500_000)
    assert result["deviation_pct"] == pytest.approx(20.0)
