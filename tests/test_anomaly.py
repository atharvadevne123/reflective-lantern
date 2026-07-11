"""Tests for valuation anomaly detection."""

import pytest

from app.anomaly import detect_valuation_anomaly


def test_normal_value_not_anomaly() -> None:
    result = detect_valuation_anomaly(predicted=500_000, neighborhood_median=480_000)
    assert result["is_anomaly"] is False
    assert result["direction"] == "high"


def test_severely_underpriced_is_anomaly() -> None:
    result = detect_valuation_anomaly(predicted=100_000, neighborhood_median=800_000)
    assert result["is_anomaly"] is True
    assert result["direction"] == "low"


def test_severely_overpriced_is_anomaly() -> None:
    result = detect_valuation_anomaly(predicted=3_000_000, neighborhood_median=400_000)
    assert result["is_anomaly"] is True
    assert result["direction"] == "high"


def test_zscore_method_used_when_std_provided() -> None:
    result = detect_valuation_anomaly(
        predicted=2_000_000,
        neighborhood_median=500_000,
        neighborhood_std=50_000,
    )
    assert result["method"] == "zscore"
    assert result["is_anomaly"] is True


def test_zscore_within_threshold_not_anomaly() -> None:
    result = detect_valuation_anomaly(
        predicted=510_000,
        neighborhood_median=500_000,
        neighborhood_std=50_000,
    )
    assert result["method"] == "zscore"
    assert result["is_anomaly"] is False


def test_iqr_method_used_with_reference_values() -> None:
    ref = list(range(300_000, 600_000, 10_000))
    result = detect_valuation_anomaly(
        predicted=5_000_000,
        neighborhood_median=450_000,
        reference_values=ref,
    )
    assert result["method"] == "iqr"
    assert result["is_anomaly"] is True


def test_zero_predicted_skipped() -> None:
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
def test_ratio_method_parametrized(predicted, median, expect_anomaly) -> None:
    result = detect_valuation_anomaly(predicted=predicted, neighborhood_median=median)
    assert result["is_anomaly"] == expect_anomaly


def test_deviation_pct_computed() -> None:
    result = detect_valuation_anomaly(predicted=600_000, neighborhood_median=500_000)
    assert result["deviation_pct"] == pytest.approx(20.0)


def test_result_has_required_keys() -> None:
    result = detect_valuation_anomaly(predicted=450_000, neighborhood_median=500_000)
    assert "is_anomaly" in result
    assert "direction" in result
    assert "method" in result
    assert "deviation_pct" in result


def test_direction_low_when_underpriced() -> None:
    result = detect_valuation_anomaly(predicted=50_000, neighborhood_median=500_000)
    assert result["direction"] == "low"


def test_direction_high_when_overpriced() -> None:
    result = detect_valuation_anomaly(predicted=1_500_000, neighborhood_median=500_000)
    assert result["direction"] == "high"


@pytest.mark.parametrize("std", [10_000, 50_000, 100_000])
def test_zscore_deviation_scales_with_std(std) -> None:
    result = detect_valuation_anomaly(
        predicted=700_000, neighborhood_median=500_000, neighborhood_std=std
    )
    assert result["method"] == "zscore"
    assert "score" in result


@pytest.mark.parametrize(
    "predicted,median,std,expected_anomaly",
    [
        (500_000, 500_000, 50_000, False),
        (700_000, 500_000, 20_000, True),
        (400_000, 500_000, 200_000, False),
    ],
)
def test_zscore_parametrized(predicted, median, std, expected_anomaly) -> None:
    result = detect_valuation_anomaly(
        predicted=predicted, neighborhood_median=median, neighborhood_std=std
    )
    assert result["is_anomaly"] == expected_anomaly


def test_iqr_normal_value_not_anomaly() -> None:
    ref = list(range(400_000, 600_000, 5_000))
    result = detect_valuation_anomaly(
        predicted=490_000, neighborhood_median=500_000, reference_values=ref
    )
    assert result["method"] == "iqr"
    assert result["is_anomaly"] is False


def test_negative_predicted_skipped() -> None:
    result = detect_valuation_anomaly(predicted=-100_000, neighborhood_median=500_000)
    assert result["method"] == "skipped"


def test_score_field_present_in_zscore_result() -> None:
    result = detect_valuation_anomaly(
        predicted=750_000, neighborhood_median=500_000, neighborhood_std=50_000
    )
    assert "score" in result
    assert isinstance(result["score"], float)


def test_ratio_low_threshold_boundary() -> None:
    from app.anomaly import RATIO_LOW_THRESHOLD

    at_threshold = detect_valuation_anomaly(
        predicted=500_000 * RATIO_LOW_THRESHOLD,
        neighborhood_median=500_000,
    )
    assert at_threshold["is_anomaly"] is True


def test_ratio_high_threshold_boundary() -> None:
    from app.anomaly import RATIO_HIGH_THRESHOLD

    just_above = detect_valuation_anomaly(
        predicted=500_000 * (RATIO_HIGH_THRESHOLD + 0.1),
        neighborhood_median=500_000,
    )
    assert just_above["is_anomaly"] is True


def test_iqr_method_requires_min_reference_size() -> None:
    from app.anomaly import MIN_REFERENCE_SIZE

    small_ref = [400_000.0] * (MIN_REFERENCE_SIZE - 1)
    result = detect_valuation_anomaly(
        predicted=5_000_000,
        neighborhood_median=450_000,
        reference_values=small_ref,
    )
    assert result["method"] != "iqr"


@pytest.mark.parametrize("deviation_pct", [-50.0, 0.0, 50.0, 200.0])
def test_deviation_pct_formula(deviation_pct: float) -> None:
    median = 500_000
    predicted = median * (1 + deviation_pct / 100)
    result = detect_valuation_anomaly(predicted=predicted, neighborhood_median=median)
    assert abs(result["deviation_pct"] - deviation_pct) < 0.1
