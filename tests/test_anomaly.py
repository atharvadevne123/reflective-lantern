"""Extended anomaly analysis tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.anomaly import compute_severity, iqr_flag, zscore_flag


def test_zscore_normal():
    assert not zscore_flag(10.0, mean=10.0, std=2.0, threshold=3.0)


def test_zscore_outlier():
    assert zscore_flag(20.0, mean=10.0, std=1.0, threshold=3.0)


def test_zscore_zero_std():
    assert not zscore_flag(5.0, mean=5.0, std=0.0)

    def test_insufficient_reference(self):
        result = quick_anomaly_check([3000.0] * 5, [3000.0, 9000.0])
        assert "error" in result

def test_iqr_within_fence():
    assert not iqr_flag(10.0, q1=8.0, q3=12.0)


def test_iqr_outside_fence():
    assert iqr_flag(30.0, q1=8.0, q3=12.0)


def test_iqr_below_fence():
    assert iqr_flag(-5.0, q1=8.0, q3=12.0)


def test_compute_severity_none():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(10.0, ref)
    assert result["severity"] == "none"


def test_compute_severity_critical():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(100.0, ref)
    assert result["severity"] == "critical"
    assert result["z_flag"] is True
    assert result["iqr_flag"] is True


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


def test_ratio_low_threshold_is_less_than_one() -> None:
    from app.anomaly import RATIO_LOW_THRESHOLD

    assert RATIO_LOW_THRESHOLD < 1.0


def test_ratio_high_threshold_is_greater_than_one() -> None:
    from app.anomaly import RATIO_HIGH_THRESHOLD

    assert RATIO_HIGH_THRESHOLD > 1.0


def test_min_reference_size_at_least_two() -> None:
    from app.anomaly import MIN_REFERENCE_SIZE

    assert MIN_REFERENCE_SIZE >= 2


@pytest.mark.parametrize("predicted,median,expect_anomaly", [
    (450_000, 500_000, False),   # within range
    (100_000, 500_000, True),    # too low
    (2_000_000, 500_000, True),  # too high
])
def test_anomaly_detection_threshold_behaviour(predicted, median, expect_anomaly) -> None:
    result = detect_valuation_anomaly(predicted=predicted, neighborhood_median=median)
    assert result["is_anomaly"] == expect_anomaly
