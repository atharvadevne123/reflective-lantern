"""Drift detection and monitoring tests."""
from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import compute_drift, get_prediction_metrics, log_prediction


def test_compute_drift_no_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(50, 5, 100).tolist()
    cur = rng.normal(50, 5, 100).tolist()
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert isinstance(result["drift_detected"], bool)


def test_compute_drift_detects_drift():
    rng = np.random.default_rng(1)
    ref = rng.normal(50, 5, 200).tolist()
    cur = rng.normal(80, 5, 200).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.3


def test_compute_drift_insufficient_samples():
    result = compute_drift([1.0], [2.0])
    assert "error" in result
    assert result["drift_detected"] is False


@pytest.mark.parametrize("ref_mean,cur_mean,expect_drift", [
    (10, 10, False),
    (10, 100, True),
])
def test_drift_parametrised(ref_mean, cur_mean, expect_drift):
    rng = np.random.default_rng(42)
    ref = rng.normal(ref_mean, 1, 200).tolist()
    cur = rng.normal(cur_mean, 1, 200).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is expect_drift


def test_log_prediction_creates_record(db_session):
    record = log_prediction(
        db=db_session,
        ticker="AAPL",
        alpha_score=72.5,
        return_prediction=0.22,
        confidence=0.75,
    )
    assert record.id is not None
    assert record.ticker == "AAPL"
    assert record.alpha_score == 72.5


def test_get_prediction_metrics_empty(db_session):
    metrics = get_prediction_metrics(db_session)
    assert "total_predictions" in metrics


def test_ks_statistic_bounded():
    rng = np.random.default_rng(99)
    ref = rng.uniform(0, 1, 100).tolist()
    cur = rng.uniform(0, 1, 100).tolist()
    result = compute_drift(ref, cur)
    assert 0.0 <= result["ks_statistic"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0


def test_pipeline_health_check_fitted(trained_pipeline):
    from app.monitoring import check_pipeline_health
    result = check_pipeline_health(trained_pipeline)
    assert result["healthy"] is True
    assert result["issues"] == []
