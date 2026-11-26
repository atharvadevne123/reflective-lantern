"""Drift detection and monitoring tests."""

import pytest

from app.monitoring import MIN_DRIFT_SAMPLES, compute_drift


def test_compute_drift_no_drift():
    import numpy as np

    rng = np.random.default_rng(42)
    ref = rng.normal(0.7, 0.1, MIN_DRIFT_SAMPLES + 10).tolist()
    cur = rng.normal(0.7, 0.1, MIN_DRIFT_SAMPLES + 10).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is False
    assert result["ks_statistic"] is not None


def test_compute_drift_detects_shift():
    import numpy as np

    rng = np.random.default_rng(99)
    ref = rng.normal(0.8, 0.05, MIN_DRIFT_SAMPLES + 20).tolist()
    cur = rng.normal(0.3, 0.05, MIN_DRIFT_SAMPLES + 20).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["p_value"] < 0.05


def test_compute_drift_insufficient_samples():
    result = compute_drift([0.5, 0.6, 0.7], [0.4, 0.5])
    assert result["drift_detected"] is False
    assert result["reason"] == "insufficient_samples"
    assert result["ks_statistic"] is None


def test_compute_drift_returns_counts():
    import numpy as np

    rng = np.random.default_rng(1)
    n = MIN_DRIFT_SAMPLES + 5
    ref = rng.uniform(0, 1, n).tolist()
    cur = rng.uniform(0, 1, n).tolist()
    result = compute_drift(ref, cur)
    assert result["reference_n"] == n
    assert result["current_n"] == n


def test_get_drift_report_empty_db(db_session):
    from app.monitoring import get_drift_report

    report = get_drift_report(db_session)
    assert report["status"] in ("no_data", "ok")


def test_get_drift_report_with_predictions(db_session):
    from app.monitoring import get_drift_report, log_prediction

    for i in range(5):
        log_prediction(
            db_session,
            ticket_id=f"T-{i}",
            subject="VPN not working",
            description="Cannot connect to VPN from home error 800",
            prediction={
                "priority": "high",
                "confidence": 0.75 + i * 0.01,
                "resolution_hours_estimate": 8.0,
                "category": "network",
            },
        )
    report = get_drift_report(db_session)
    assert report["total_predictions"] >= 5


@pytest.mark.parametrize(
    "ref_mean,cur_mean,expect_drift",
    [
        (0.75, 0.74, False),
        (0.9, 0.1, True),
    ],
)
def test_compute_drift_parametrized(ref_mean, cur_mean, expect_drift):
    import numpy as np

    rng = np.random.default_rng(7)
    n = MIN_DRIFT_SAMPLES + 10
    ref = rng.normal(ref_mean, 0.03, n).tolist()
    cur = rng.normal(cur_mean, 0.03, n).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is expect_drift
