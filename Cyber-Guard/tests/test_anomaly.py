"""Anomaly detection tests for Cyber-Guard."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from app.anomaly import (
    batch_anomaly_rate,
    load_anomaly_detector,
    score_anomaly,
    train_anomaly_detector,
)
from app.model import generate_synthetic_data


@pytest.fixture(scope="module")
def trained_detector():
    X, _ = generate_synthetic_data(300)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "anomaly.joblib")
        train_anomaly_detector(X, model_path=path)
        yield load_anomaly_detector(path)


def test_score_anomaly_returns_expected_keys(trained_detector, sample_dataframe):
    result = score_anomaly(sample_dataframe, trained_detector)
    assert set(result) == {"anomaly_score", "decision_score", "is_anomaly"}
    assert isinstance(result["is_anomaly"], bool)


def test_typical_connection_is_not_anomalous(trained_detector):
    """A connection drawn from the training distribution should look normal."""
    X, _ = generate_synthetic_data(50)
    rate = batch_anomaly_rate(X, trained_detector)
    # Contamination is 0.05, so a same-distribution batch should stay well
    # under half flagged; a much higher rate means the detector is degenerate.
    assert rate < 0.5


def test_extreme_connection_is_flagged(trained_detector):
    """A connection orders of magnitude outside training range is an outlier."""
    extreme = pd.DataFrame([{
        "src_bytes": 5_000_000_000.0,
        "dst_bytes": 5_000_000_000.0,
        "duration": 500_000.0,
        "protocol_type": "icmp",
        "service": "other",
        "flag": "OTH",
    }])
    result = score_anomaly(extreme, trained_detector)
    assert result["is_anomaly"] is True
    assert result["anomaly_score"] > 0


def test_batch_anomaly_rate_empty(trained_detector):
    empty = pd.DataFrame(columns=[
        "src_bytes", "dst_bytes", "duration", "protocol_type", "service", "flag",
    ])
    assert batch_anomaly_rate(empty, trained_detector) == 0.0


def test_batch_anomaly_rate_in_unit_interval(trained_detector):
    X, _ = generate_synthetic_data(100)
    rate = batch_anomaly_rate(X, trained_detector)
    assert 0.0 <= rate <= 1.0


def test_anomaly_score_is_negated_decision(trained_detector, sample_dataframe):
    """anomaly_score must be the sign-flipped decision score, higher == weirder."""
    result = score_anomaly(sample_dataframe, trained_detector)
    assert result["anomaly_score"] == pytest.approx(-result["decision_score"], abs=1e-4)


def test_train_anomaly_detector_persists(tmp_path):
    X, _ = generate_synthetic_data(120)
    path = tmp_path / "anom.joblib"
    train_anomaly_detector(X, model_path=str(path))
    assert path.exists()


@pytest.mark.parametrize("contamination", [0.01, 0.1])
def test_contamination_setting_accepted(contamination: float, tmp_path):
    X, _ = generate_synthetic_data(120)
    path = tmp_path / f"anom_{contamination}.joblib"
    pipe = train_anomaly_detector(X, contamination=contamination, model_path=str(path))
    assert pipe is not None
