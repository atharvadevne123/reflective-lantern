"""Drift detection and monitoring tests."""
from __future__ import annotations

import pytest

from app.monitoring import compute_drift, compute_psi, get_drift_summary, get_prediction_metrics


def test_compute_drift_same_distribution():
    data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    result = compute_drift(data, data)
    assert result["drift_detected"] is False
    assert result["ks_statistic"] == pytest.approx(0.0)


def test_compute_drift_different_distributions():
    ref = [0.0] * 100
    cur = [10.0] * 100
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] == pytest.approx(1.0)


def test_compute_drift_insufficient_data():
    result = compute_drift([1.0], [2.0])
    assert "error" in result
    assert result["drift_detected"] is False


def test_compute_psi_identical_distributions():
    data = list(range(100))
    psi = compute_psi(
        __import__("numpy").array(data, dtype=float),
        __import__("numpy").array(data, dtype=float),
    )
    assert psi < 0.1


def test_compute_psi_shifted_distribution():
    import numpy as np

    ref = np.arange(100, dtype=float)
    cur = np.arange(100, dtype=float) + 50
    psi = compute_psi(ref, cur)
    assert psi > 0.1


@pytest.mark.parametrize("ref,cur,expected_drift", [
    ([0.0] * 50 + [1.0] * 50, [0.0] * 50 + [1.0] * 50, False),
    ([0.0] * 100, [100.0] * 100, True),
])
def test_compute_drift_parametrized(ref, cur, expected_drift):
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is expected_drift


def test_log_and_get_drift_summary(db_session):
    from app.monitoring import log_drift

    log_drift(db_session, "home_elo", 0.15, 0.03, True)
    log_drift(db_session, "away_elo", 0.05, 0.42, False)
    summary = get_drift_summary(db_session)
    assert len(summary) >= 2
    feature_names = [s["feature"] for s in summary]
    assert "home_elo" in feature_names


def test_get_prediction_metrics_empty(db_session):
    metrics = get_prediction_metrics(db_session)
    assert "total_predictions" in metrics
    assert "outcome_distribution" in metrics


def test_log_prediction_and_metrics(db_session):
    from app.monitoring import log_prediction

    log_prediction(
        db_session,
        match_id="metric-test-001",
        home_team="A", away_team="B",
        predicted_outcome="home_win",
        home_win_prob=0.6, draw_prob=0.25, away_win_prob=0.15,
        confidence=0.6,
    )
    metrics = get_prediction_metrics(db_session)
    assert metrics["total_predictions"] >= 1
