"""Tests for Isolation Forest anomaly detection."""

from __future__ import annotations

import numpy as np
import pytest

from app.anomaly import fit_isolation_forest, quick_anomaly_check, score_loads


class TestFitIsolationForest:
    def test_fits_successfully(self):
        loads = np.random.default_rng(42).normal(3000, 200, 500).tolist()
        model = fit_isolation_forest(loads)
        assert model is not None

    def test_model_has_predict(self):
        loads = np.random.default_rng(42).normal(3000, 200, 200).tolist()
        model = fit_isolation_forest(loads)
        assert hasattr(model, "predict")

    @pytest.mark.parametrize("contamination", [0.01, 0.05, 0.10])
    def test_various_contamination(self, contamination):
        loads = np.random.default_rng(0).normal(3000, 200, 300).tolist()
        model = fit_isolation_forest(loads, contamination=contamination)
        assert model is not None


class TestScoreLoads:
    def test_returns_list_of_dicts(self):
        loads = np.random.default_rng(42).normal(3000, 200, 200).tolist()
        model = fit_isolation_forest(loads)
        results = score_loads(model, loads[:10])
        assert len(results) == 10
        assert all("is_anomaly" in r for r in results)
        assert all("anomaly_score" in r for r in results)

    def test_obvious_anomaly_flagged(self):
        rng = np.random.default_rng(7)
        loads = rng.normal(3000, 100, 300).tolist()
        model = fit_isolation_forest(loads, contamination=0.05)
        outliers = [99999.0, -5000.0]
        results = score_loads(model, outliers)
        assert any(r["is_anomaly"] for r in results)

    def test_index_matches_input(self):
        loads = [3000.0] * 100
        model = fit_isolation_forest(loads)
        results = score_loads(model, loads[:5])
        for i, r in enumerate(results):
            assert r["index"] == i


class TestQuickAnomalyCheck:
    def test_returns_summary_dict(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(3000, 200, 200).tolist()
        new = rng.normal(3000, 200, 20).tolist()
        result = quick_anomaly_check(ref, new)
        assert "anomalies_detected" in result
        assert "anomaly_rate_pct" in result
        assert result["total_checked"] == 20

    def test_insufficient_reference(self):
        result = quick_anomaly_check([3000.0] * 5, [3000.0, 9000.0])
        assert "error" in result

    def test_high_anomaly_rate_for_outliers(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(3000, 100, 300).tolist()
        outliers = [99999.0] * 10
        result = quick_anomaly_check(ref, outliers)
        assert result["anomaly_rate_pct"] > 50
