"""Drift detection and monitoring tests."""
from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import compute_drift, log_prediction


def test_compute_drift_no_drift():
    rng = np.random.default_rng(0)
    result = compute_drift(rng.normal(50, 5, 100).tolist(), rng.normal(50, 5, 100).tolist())
    assert isinstance(result["drift_detected"], bool)


def test_compute_drift_detects_drift():
    rng = np.random.default_rng(1)
    result = compute_drift(rng.normal(50, 5, 200).tolist(), rng.normal(80, 5, 200).tolist())
    assert result["drift_detected"] is True


def test_compute_drift_insufficient_samples():
    assert "error" in compute_drift([1.0], [2.0])


def test_log_prediction_creates_record(db_session):
    record = log_prediction(db=db_session, ticker="AAPL", alpha_score=72.5, return_prediction=0.22, confidence=0.75)
    assert record.ticker == "AAPL"
