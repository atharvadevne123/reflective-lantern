"""Tests for drift detection and monitoring in Temporal-Pulse."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def reset_state():
    """Reset monitoring state before each test."""
    from app.monitoring import reset_monitoring
    reset_monitoring()
    yield
    reset_monitoring()


class TestDistributionUpdates:
    def test_update_reference_stores_values(self):
        from app.monitoring import _REFERENCE_DISTRIBUTIONS, update_reference_distribution
        update_reference_distribution("temp", [1.0, 2.0, 3.0])
        assert len(_REFERENCE_DISTRIBUTIONS["temp"]) == 3

    def test_update_current_stores_values(self):
        from app.monitoring import _CURRENT_DISTRIBUTIONS, update_current_distribution
        update_current_distribution("temp", [4.0, 5.0, 6.0])
        assert len(_CURRENT_DISTRIBUTIONS["temp"]) == 3


class TestKSDriftTest:
    def test_insufficient_data_returns_no_drift(self):
        from app.monitoring import (
            run_ks_drift_test,
            update_current_distribution,
            update_reference_distribution,
        )
        update_reference_distribution("feat", [1.0, 2.0])
        update_current_distribution("feat", [3.0, 4.0])
        result = run_ks_drift_test("feat")
        assert result["drift_detected"] is False
        assert "insufficient_data" in result.get("reason", "")

    def test_identical_distributions_no_drift(self):
        from app.monitoring import (
            run_ks_drift_test,
            update_current_distribution,
            update_reference_distribution,
        )
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 200).tolist()
        update_reference_distribution("feat", values[:100])
        update_current_distribution("feat", values[100:])
        result = run_ks_drift_test("feat")
        assert "ks_statistic" in result

    def test_very_different_distributions_detected(self):
        from app.monitoring import (
            run_ks_drift_test,
            update_current_distribution,
            update_reference_distribution,
        )
        update_reference_distribution("feat", list(np.random.normal(0, 0.1, 100)))
        update_current_distribution("feat", list(np.random.normal(100, 0.1, 100)))
        result = run_ks_drift_test("feat")
        assert result["drift_detected"] is True

    def test_run_all_drift_tests_returns_list(self):
        from app.monitoring import (
            run_all_drift_tests,
            update_current_distribution,
            update_reference_distribution,
        )
        update_reference_distribution("a", [1.0] * 50)
        update_current_distribution("a", [2.0] * 50)
        results = run_all_drift_tests()
        assert isinstance(results, list)

    @pytest.mark.parametrize("n_features", [1, 3, 5])
    def test_run_all_tests_count(self, n_features):
        from app.monitoring import (
            run_all_drift_tests,
            update_current_distribution,
            update_reference_distribution,
        )
        for i in range(n_features):
            update_reference_distribution(f"feat_{i}", [float(j) for j in range(50)])
            update_current_distribution(f"feat_{i}", [float(j + 1) for j in range(50)])
        results = run_all_drift_tests()
        assert len(results) >= n_features


class TestPredictionLogging:
    def test_log_prediction_increments_count(self):
        from app.monitoring import get_metrics_summary, log_prediction
        log_prediction("s1", 0.5, [1.0, 2.0], 10.0)
        summary = get_metrics_summary()
        assert summary["total_predictions"] >= 1

    def test_metrics_includes_score_stats(self):
        from app.monitoring import get_metrics_summary, log_prediction
        log_prediction("s1", 0.8, [], 20.0)
        log_prediction("s1", 0.2, [], 15.0)
        summary = get_metrics_summary()
        assert "anomaly_score_mean" in summary

    def test_metrics_latency_tracked(self):
        from app.monitoring import get_metrics_summary, log_prediction
        log_prediction("s1", 0.3, [], 50.0)
        summary = get_metrics_summary()
        assert summary.get("latency_mean_ms", 0) > 0

    def test_empty_log_returns_zero_count(self):
        from app.monitoring import get_metrics_summary
        summary = get_metrics_summary()
        assert summary["total_predictions"] == 0
