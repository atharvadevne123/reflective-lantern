"""Extended tests for Cyber-Guard app/anomaly.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.anomaly import DEFAULT_CONTAMINATION, batch_anomaly_rate, score_anomaly, train_anomaly_detector
from app.features import make_sample_df


def _training_df(n: int = 100) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        rows.append({
            "src_bytes": np.random.rand() * 1000,
            "dst_bytes": np.random.rand() * 1000,
            "duration": np.random.rand() * 5,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
        })
    return pd.DataFrame(rows)


class TestDefaultContamination:
    def test_is_float(self) -> None:
        assert isinstance(DEFAULT_CONTAMINATION, float)

    def test_in_valid_range(self) -> None:
        assert 0.0 < DEFAULT_CONTAMINATION < 0.5


class TestTrainAnomalyDetector:
    def test_returns_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline
        X = _training_df(50)
        pipe = train_anomaly_detector(X, model_path="/dev/null")
        assert hasattr(pipe, "predict")

    def test_trained_can_predict(self) -> None:
        X = _training_df(50)
        pipe = train_anomaly_detector(X, model_path="/dev/null")
        sample = make_sample_df()
        result = pipe.predict(sample)
        assert result[0] in (-1, 1)


class TestScoreAnomaly:
    def _fitted_pipe(self):
        X = _training_df(100)
        return train_anomaly_detector(X, model_path="/dev/null")

    def test_returns_dict(self) -> None:
        pipe = self._fitted_pipe()
        result = score_anomaly(make_sample_df(), pipe)
        assert isinstance(result, dict)

    def test_has_is_anomaly(self) -> None:
        pipe = self._fitted_pipe()
        result = score_anomaly(make_sample_df(), pipe)
        assert "is_anomaly" in result

    def test_has_anomaly_score(self) -> None:
        pipe = self._fitted_pipe()
        result = score_anomaly(make_sample_df(), pipe)
        assert "anomaly_score" in result

    def test_is_anomaly_is_bool(self) -> None:
        pipe = self._fitted_pipe()
        result = score_anomaly(make_sample_df(), pipe)
        assert isinstance(result["is_anomaly"], bool)


class TestBatchAnomalyRate:
    def _fitted_pipe(self):
        X = _training_df(100)
        return train_anomaly_detector(X, model_path="/dev/null")

    def test_empty_df_returns_zero(self) -> None:
        pipe = self._fitted_pipe()
        empty = pd.DataFrame(columns=["src_bytes", "dst_bytes", "duration", "protocol_type", "service", "flag"])
        result = batch_anomaly_rate(empty, pipe)
        assert result == pytest.approx(0.0)

    def test_returns_float_in_range(self) -> None:
        pipe = self._fitted_pipe()
        X = _training_df(50)
        rate = batch_anomaly_rate(X, pipe)
        assert 0.0 <= rate <= 1.0
