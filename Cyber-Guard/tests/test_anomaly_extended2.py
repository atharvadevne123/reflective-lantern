"""Extended anomaly detection tests with parametrize and boundary coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.anomaly import (
    batch_anomaly_rate,
    score_anomaly,
    train_anomaly_detector,
)
from app.model import generate_synthetic_data


def _make_training_data(n: int = 200) -> pd.DataFrame:
    X, _ = generate_synthetic_data(n)
    return X


def test_train_anomaly_detector_returns_pipeline(tmp_path) -> None:
    X = _make_training_data(100)
    path = str(tmp_path / "anomaly.joblib")
    pipe = train_anomaly_detector(X, model_path=path)
    assert isinstance(pipe, Pipeline)


def test_train_anomaly_detector_pipeline_has_detector_step(tmp_path) -> None:
    X = _make_training_data(100)
    path = str(tmp_path / "anomaly.joblib")
    pipe = train_anomaly_detector(X, model_path=path)
    assert "detector" in pipe.named_steps


def test_score_anomaly_has_required_keys(tmp_path) -> None:
    X = _make_training_data(100)
    path = str(tmp_path / "anomaly.joblib")
    pipe = train_anomaly_detector(X, model_path=path)
    sample = X.iloc[:1].copy()
    result = score_anomaly(sample, pipe)
    for key in ("anomaly_score", "decision_score", "is_anomaly"):
        assert key in result


def test_score_anomaly_is_anomaly_is_bool(tmp_path) -> None:
    X = _make_training_data(100)
    path = str(tmp_path / "anomaly.joblib")
    pipe = train_anomaly_detector(X, model_path=path)
    sample = X.iloc[:1].copy()
    result = score_anomaly(sample, pipe)
    assert isinstance(result["is_anomaly"], bool)


def test_batch_anomaly_rate_empty_dataframe_returns_zero(tmp_path) -> None:
    X = _make_training_data(100)
    path = str(tmp_path / "anomaly.joblib")
    pipe = train_anomaly_detector(X, model_path=path)
    empty = X.iloc[:0].copy()
    assert batch_anomaly_rate(empty, pipe) == 0.0


def test_batch_anomaly_rate_in_zero_one(tmp_path) -> None:
    X = _make_training_data(200)
    path = str(tmp_path / "anomaly.joblib")
    pipe = train_anomaly_detector(X, model_path=path)
    rate = batch_anomaly_rate(X.iloc[:50], pipe)
    assert 0.0 <= rate <= 1.0


@pytest.mark.parametrize("contamination", [0.05, 0.10, 0.20])
def test_train_anomaly_detector_contamination_parametrized(contamination: float, tmp_path) -> None:
    X = _make_training_data(200)
    path = str(tmp_path / f"anomaly_{contamination:.0%}.joblib")
    pipe = train_anomaly_detector(X, contamination=contamination, model_path=path)
    preds = pipe.predict(X)
    anomaly_rate = np.mean(preds == -1)
    assert anomaly_rate == pytest.approx(contamination, abs=0.1)


@pytest.mark.parametrize("n_train", [50, 100, 200])
def test_train_anomaly_detector_various_sizes(n_train: int, tmp_path) -> None:
    X = _make_training_data(n_train)
    path = str(tmp_path / f"anomaly_{n_train}.joblib")
    pipe = train_anomaly_detector(X, model_path=path)
    sample = X.iloc[:5].copy()
    preds = pipe.predict(sample)
    assert len(preds) == 5
