"""Tests for monitoring and drift detection."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    PredictionTracker,
    compute_all_feature_drifts,
    compute_drift,
    compute_prediction_error_metrics,
    get_tracker,
)


class TestComputeDrift:
    def test_no_drift_identical_distributions(self):
        ref = list(np.random.default_rng(42).normal(0, 1, 500))
        cur = list(np.random.default_rng(42).normal(0, 1, 200))
        result = compute_drift(ref, cur, "test_feature")
        assert result["drift_detected"] is False
        assert result["p_value"] > 0.05

    def test_drift_detected_shifted_distribution(self):
        ref = list(np.random.default_rng(42).normal(0, 1, 500))
        cur = list(np.random.default_rng(42).normal(5, 1, 200))
        result = compute_drift(ref, cur, "shifted_feature")
        assert result["drift_detected"] is True
        assert result["p_value"] < 0.05

    def test_returns_expected_keys(self):
        ref = list(np.random.default_rng(0).uniform(0, 1, 100))
        cur = list(np.random.default_rng(1).uniform(0, 1, 50))
        result = compute_drift(ref, cur, "feature_x")
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "drift_detected" in result
        assert "feature" in result

    def test_insufficient_samples_no_drift(self):
        result = compute_drift([1.0, 2.0], [3.0], "tiny")
        assert result["drift_detected"] is False
        assert result["note"] == "insufficient_samples"

    def test_ks_statistic_in_range(self):
        ref = list(np.random.default_rng(7).normal(100, 15, 300))
        cur = list(np.random.default_rng(8).normal(110, 15, 100))
        result = compute_drift(ref, cur, "load_mw")
        assert 0.0 <= result["ks_statistic"] <= 1.0

    @pytest.mark.parametrize("shift", [0, 1, 3, 10])
    def test_drift_increases_with_shift(self, shift):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500).tolist()
        cur = rng.normal(shift, 1, 200).tolist()
        result = compute_drift(ref, cur, f"shift_{shift}")
        if shift == 0:
            assert result["drift_detected"] is False
        elif shift >= 10:
            assert result["drift_detected"] is True


class TestComputeAllFeatureDrifts:
    def test_runs_on_dataframe(self):
        import pandas as pd
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(50, 10, 300)})
        cur = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(50, 10, 100)})
        results = compute_all_feature_drifts(ref, cur, ["a", "b"])
        assert len(results) == 2
        assert all("feature" in r for r in results)


class TestPredictionErrorMetrics:
    def test_perfect_prediction(self):
        vals = [1000.0, 2000.0, 3000.0]
        m = compute_prediction_error_metrics(vals, vals)
        assert m["rmse"] == pytest.approx(0.0, abs=0.001)
        assert m["mae"] == pytest.approx(0.0, abs=0.001)

    def test_known_rmse(self):
        actual = [3.0, 3.0]
        predicted = [4.0, 4.0]
        m = compute_prediction_error_metrics(actual, predicted)
        assert m["rmse"] == pytest.approx(1.0, abs=0.001)

    def test_empty_inputs(self):
        m = compute_prediction_error_metrics([], [])
        assert m["rmse"] is None

    def test_mismatched_lengths(self):
        m = compute_prediction_error_metrics([1.0, 2.0], [1.0])
        assert m["rmse"] is None

    def test_mape_zero_actual(self):
        m = compute_prediction_error_metrics([0.0, 1.0], [0.5, 1.5])
        assert m["mape"] is not None


class TestPredictionTracker:
    def test_records_loads(self):
        t = PredictionTracker(max_size=10)
        t.record(3000.0, 5.0)
        t.record(3500.0, 6.0)
        assert t.count == 2
        assert t.loads == [3000.0, 3500.0]

    def test_evicts_oldest_when_full(self):
        t = PredictionTracker(max_size=3)
        for i in range(5):
            t.record(float(i), 1.0)
        assert t.count == 3
        assert t.loads[0] == 2.0  # oldest 0,1 evicted

    def test_get_tracker_returns_singleton(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2
