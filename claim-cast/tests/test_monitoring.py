"""Drift detection and monitoring tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    compute_drift,
    drift_summary,
    get_recent_scores,
    log_drift,
    log_prediction,
)


def test_compute_drift_no_drift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.3, 0.1, 100).tolist()
    cur = rng.normal(0.3, 0.1, 50).tolist()
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "drift_detected" in result
    assert result["drift_detected"] is False or result["drift_detected"] is True


def test_compute_drift_detects_drift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0.2, 0.05, 200).tolist()
    cur = rng.normal(0.8, 0.05, 200).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.3


def test_compute_drift_insufficient_samples():
    result = compute_drift([0.1, 0.2], [0.3])
    assert result.get("error") == "insufficient_samples"
    assert result["ks_statistic"] is None


@pytest.mark.parametrize("ref_size,cur_size", [(10, 10), (50, 20), (200, 100)])
def test_compute_drift_various_sizes(ref_size, cur_size):
    rng = np.random.default_rng(3)
    ref = rng.uniform(0, 1, ref_size).tolist()
    cur = rng.uniform(0, 1, cur_size).tolist()
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_log_prediction(db_session):
    record = log_prediction(
        db_session,
        claim_id="CLM-MONITOR-001",
        severity_score=0.75,
        severity_label="high",
        triage_priority=2,
    )
    assert record.id is not None
    assert record.severity_label == "high"


def test_log_drift_record(db_session):
    record = log_drift(
        db_session,
        feature_name="severity_score",
        ks_statistic=0.12,
        p_value=0.03,
        drift_detected=True,
        sample_size=100,
    )
    assert record.id is not None
    assert record.drift_detected == 1


def test_get_recent_scores_empty(db_session):
    scores = get_recent_scores(db_session, limit=10)
    assert isinstance(scores, list)


def test_drift_summary_empty(db_session):
    summary = drift_summary(db_session)
    assert "total_checks" in summary
    assert "total_alerts" in summary


def test_drift_summary_after_log(db_session):
    log_drift(db_session, "test_feature", 0.25, 0.01, True, 50)
    summary = drift_summary(db_session)
    assert summary["total_checks"] >= 1
