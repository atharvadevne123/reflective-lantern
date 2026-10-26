"""Tests for drift detection and prediction logging."""

import numpy as np
import pytest

from app.monitoring import (
    compute_drift,
    get_drift_summary,
    log_prediction,
    run_full_drift_check,
)


def test_compute_drift_no_drift() -> None:
    # Two samples drawn from the same distribution must not trip the KS test.
    rng = np.random.default_rng(7)
    ref = rng.normal(loc=10.0, scale=2.0, size=200).tolist()
    cur = rng.normal(loc=10.0, scale=2.0, size=200).tolist()
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert not result["drift_detected"]


def test_compute_drift_with_drift() -> None:
    ref = [0.0] * 100
    cur = [100.0] * 100
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_empty_lists() -> None:
    result = compute_drift([], [])
    assert result["drift_detected"] is False
    assert result["p_value"] == 1.0


def test_log_prediction_creates_record(db_session) -> None:
    flow = {
        "src_bytes": 500.0,
        "dst_bytes": 1000.0,
        "duration": 2.0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
    }
    result = {"predicted_class": "normal", "confidence": 0.95, "is_attack": 0}
    record = log_prediction(db_session, "test-corr-id", flow, result)
    assert record.id is not None
    assert record.predicted_class == "normal"
    assert record.confidence == 0.95
    assert record.correlation_id == "test-corr-id"


def test_log_prediction_attack_record(db_session) -> None:
    flow = {"src_bytes": 0.0, "dst_bytes": 0.0, "duration": 0.0}
    result = {"predicted_class": "dos", "confidence": 0.87, "is_attack": 1}
    record = log_prediction(db_session, "atk-corr-id", flow, result)
    assert record.is_attack == 1
    assert record.predicted_class == "dos"


def test_run_full_drift_check_empty_db(db_session) -> None:
    reference = {"src_bytes": [100.0] * 50, "dst_bytes": [200.0] * 50}
    # Empty DB — should return empty list since no recent predictions
    results = run_full_drift_check(db_session, reference)
    # Results depend on whether there are prediction logs; just check it doesn't crash
    assert isinstance(results, list)


def test_get_drift_summary_returns_list(db_session) -> None:
    summary = get_drift_summary(db_session, limit=10)
    assert isinstance(summary, list)


@pytest.mark.parametrize(
    "ref,cur,expect_drift",
    [
        ([1] * 50, [1] * 50, False),  # identical → no drift
        ([0] * 50, [1000] * 50, True),  # very different → drift
        ([1, 2] * 25, [1, 2] * 25, False),  # same alternating → no drift
    ],
)
def test_compute_drift_parametrize(ref: list[float], cur: list[float], expect_drift: bool) -> None:
    result = compute_drift(ref, cur)
    if expect_drift:
        assert result["drift_detected"] is True
    else:
        assert result["drift_detected"] is False


def test_compute_drift_ks_statistic_range() -> None:
    ref = list(range(100))
    cur = [x + 50 for x in range(100)]
    result = compute_drift(ref, cur)
    assert 0.0 <= result["ks_statistic"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0
