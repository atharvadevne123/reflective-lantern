"""Drift detection and monitoring tests for Risk-Sense."""

from __future__ import annotations

import pytest

from app.monitoring import (
    compute_population_stability_index,
    drift_severity,
    ks_drift_test,
    rolling_default_rate,
    summarise_feature_drift,
)


class TestKsDriftTest:
    def test_identical_distributions_no_drift(self) -> None:
        ref = [0.1 * i for i in range(20)]
        result = ks_drift_test(ref, ref[:])
        assert result["drift_detected"] is False

    def test_very_different_distributions_drift(self) -> None:
        ref = [0.0] * 30
        cur = [1.0] * 30
        result = ks_drift_test(ref, cur)
        assert result["drift_detected"] is True

    def test_insufficient_data_returns_no_drift(self) -> None:
        result = ks_drift_test([0.1, 0.2], [0.3, 0.4])
        assert result["drift_detected"] is False
        assert result.get("reason") == "insufficient_data"

    def test_result_contains_required_keys(self) -> None:
        ref = [i * 0.05 for i in range(20)]
        cur = [i * 0.05 + 0.1 for i in range(20)]
        result = ks_drift_test(ref, cur)
        assert "drift_detected" in result
        assert "ks_statistic" in result
        assert "p_value" in result

    def test_p_value_between_0_and_1(self) -> None:
        ref = [i * 0.05 for i in range(20)]
        cur = [i * 0.05 for i in range(20)]
        result = ks_drift_test(ref, cur)
        assert 0.0 <= result["p_value"] <= 1.0


class TestDriftSeverity:
    @pytest.mark.parametrize(
        "stat,expected",
        [
            (0.0, "none"),
            (0.09, "none"),
            (0.1, "mild"),
            (0.24, "mild"),
            (0.25, "moderate"),
            (0.49, "moderate"),
            (0.5, "severe"),
            (1.0, "severe"),
        ],
    )
    def test_severity_boundaries(self, stat: float, expected: str) -> None:
        assert drift_severity(stat) == expected


class TestComputePsi:
    def test_identical_distributions_psi_near_zero(self) -> None:
        ref = [i * 0.1 for i in range(50)]
        psi = compute_population_stability_index(ref, ref[:])
        assert psi < 0.1

    def test_opposite_distributions_high_psi(self) -> None:
        ref = [0.0] * 50
        cur = [1.0] * 50
        psi = compute_population_stability_index(ref, cur)
        assert psi > 0.0

    def test_constant_values_returns_zero(self) -> None:
        ref = [0.5] * 20
        cur = [0.5] * 20
        psi = compute_population_stability_index(ref, cur)
        assert psi == pytest.approx(0.0)

    def test_custom_bins(self) -> None:
        ref = [i * 0.1 for i in range(50)]
        cur = [i * 0.1 + 0.05 for i in range(50)]
        psi_5 = compute_population_stability_index(ref, cur, bins=5)
        psi_20 = compute_population_stability_index(ref, cur, bins=20)
        assert isinstance(psi_5, float)
        assert isinstance(psi_20, float)


class TestRollingDefaultRate:
    def test_empty_returns_empty(self) -> None:
        assert rolling_default_rate([]) == []

    def test_single_element(self) -> None:
        result = rolling_default_rate([0.7])
        assert result == pytest.approx([0.7])

    def test_window_size_respected(self) -> None:
        predictions = [1.0] * 5 + [0.0] * 5
        result = rolling_default_rate(predictions, window=3)
        assert len(result) == 10
        assert result[4] == pytest.approx(1.0)
        assert result[9] == pytest.approx(0.0)

    def test_all_values_between_0_and_1(self) -> None:
        import random

        predictions = [random.random() for _ in range(50)]
        result = rolling_default_rate(predictions, window=10)
        assert all(0.0 <= v <= 1.0 for v in result)


class TestSummariseFeatureDrift:
    def test_returns_one_entry_per_feature(self) -> None:
        ref = {"dti": [0.3] * 20, "util": [0.5] * 20}
        cur = {"dti": [0.3] * 20, "util": [0.5] * 20}
        report = summarise_feature_drift(ref, cur)
        assert len(report) == 2

    def test_each_entry_has_feature_key(self) -> None:
        ref = {"dti": [i * 0.05 for i in range(20)]}
        cur = {"dti": [i * 0.05 for i in range(20)]}
        report = summarise_feature_drift(ref, cur)
        assert "feature" in report[0]

    def test_missing_feature_in_current_skipped(self) -> None:
        ref = {"dti": [0.3] * 20, "util": [0.5] * 20}
        cur = {"dti": [0.3] * 20}
        report = summarise_feature_drift(ref, cur)
        assert len(report) == 1
