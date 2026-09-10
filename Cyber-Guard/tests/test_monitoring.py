"""Drift detection and monitoring tests for Cyber-Guard."""

from __future__ import annotations

import pytest

from app.monitoring import compute_drift, get_prediction_stats, log_prediction


def test_compute_drift_no_drift():
    ref = list(range(50))
    cur = list(range(50))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is False
    assert result["ks_statistic"] == 0.0


def test_compute_drift_detects_shift():
    ref = [float(i) for i in range(100)]
    cur = [float(i + 500) for i in range(100)]
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.0


def test_compute_drift_insufficient_data():
    result = compute_drift([], [])
    assert "error" in result
    assert result["drift_detected"] is False


def test_compute_drift_p_value_range():
    ref = [float(i) for i in range(200)]
    cur = [float(i) * 2 for i in range(200)]
    result = compute_drift(ref, cur)
    assert 0.0 <= result["p_value"] <= 1.0


def test_log_prediction(db_session):
    features = {
        "src_bytes": 100.0,
        "dst_bytes": 50.0,
        "duration": 1.0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
    }
    record = log_prediction(db_session, features, "normal", 0.95)
    assert record.id is not None
    assert record.prediction == "normal"
    assert record.confidence == 0.95


def test_get_prediction_stats_empty(db_session):
    stats = get_prediction_stats(db_session, hours=1)
    assert stats["total"] == 0
    assert stats["class_counts"] == {}


def test_get_prediction_stats_with_data(db_session):
    features = {
        "src_bytes": 200.0,
        "dst_bytes": 100.0,
        "duration": 0.5,
        "protocol_type": "udp",
        "service": "dns",
        "flag": "SF",
    }
    log_prediction(db_session, features, "probe", 0.80)
    log_prediction(db_session, features, "normal", 0.90)
    stats = get_prediction_stats(db_session, hours=24)
    assert stats["total"] >= 2


@pytest.mark.parametrize(
    "label,conf",
    [
        ("normal", 0.99),
        ("dos", 0.85),
        ("probe", 0.72),
        ("r2l", 0.65),
        ("u2r", 0.91),
    ],
)
def test_log_prediction_all_labels(db_session, label: str, conf: float):
    features = {
        "src_bytes": 50.0,
        "dst_bytes": 25.0,
        "duration": 0.1,
        "protocol_type": "tcp",
        "service": "ftp",
        "flag": "REJ",
    }
    record = log_prediction(db_session, features, label, conf)
    assert record.prediction == label
    assert record.confidence == conf
