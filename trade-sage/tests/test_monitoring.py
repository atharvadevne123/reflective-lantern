"""Tests for drift detection and monitoring utilities."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    _reference_store,
    check_drift_all,
    compute_drift,
    log_prediction,
    update_reference,
)


def test_compute_drift_no_drift():
    # Use identical distributions to guarantee p > 0.05
    rng = np.random.default_rng(99)
    ref = rng.normal(5.0, 0.5, 500).tolist()
    curr = rng.normal(5.0, 0.5, 500).tolist()
    result = compute_drift(ref, curr)
    assert not result["drift_detected"]
    assert 0.0 <= result["p_value"] <= 1.0


def test_compute_drift_detects_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 200).tolist()
    curr = rng.normal(10, 1, 100).tolist()  # large mean shift
    result = compute_drift(ref, curr)
    assert result["drift_detected"]
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data():
    result = compute_drift([1.0, 2.0], [3.0])
    assert not result["drift_detected"]
    assert result["p_value"] == 1.0


def test_update_reference_appends():
    _reference_store.clear()
    update_reference("test_feat", [1.0, 2.0, 3.0])
    assert len(_reference_store["test_feat"]) == 3


def test_update_reference_caps_at_window():
    from app.monitoring import REFERENCE_WINDOW

    _reference_store.clear()
    big = [float(i) for i in range(REFERENCE_WINDOW + 100)]
    update_reference("cap_test", big)
    assert len(_reference_store["cap_test"]) == REFERENCE_WINDOW


def test_check_drift_all_returns_per_feature():
    _reference_store.clear()
    rng = np.random.default_rng(2)
    features = rng.standard_normal((50, 3))
    # seed reference
    check_drift_all(rng.standard_normal((200, 3)), ["f1", "f2", "f3"])
    results = check_drift_all(features, ["f1", "f2", "f3"])
    assert set(results.keys()) == {"f1", "f2", "f3"}


def test_log_prediction_persists(db_session):
    log_prediction(db_session, "AAPL", 1, 0.82, 5)
    from app.database import PredictionLog

    rows = db_session.query(PredictionLog).filter_by(ticker="AAPL").all()
    assert len(rows) >= 1
    assert rows[-1].prediction == 1


def test_log_prediction_confidence_stored(db_session):
    log_prediction(db_session, "GOOG", 0, 0.65, 10)
    from app.database import PredictionLog

    row = db_session.query(PredictionLog).filter_by(ticker="GOOG").first()
    assert row is not None
    assert abs(row.confidence - 0.65) < 0.01


@pytest.mark.parametrize("conf", [0.51, 0.75, 0.99])
def test_log_prediction_various_confidence(db_session, conf):
    log_prediction(db_session, "TSLA", 1, conf, 5)
    from app.database import PredictionLog

    row = (
        db_session.query(PredictionLog)
        .filter_by(ticker="TSLA")
        .order_by(PredictionLog.id.desc())
        .first()
    )
    assert abs(row.confidence - round(conf, 4)) < 0.001
