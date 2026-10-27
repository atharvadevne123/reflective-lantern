"""Tests for sport-cast app/mlflow_stub.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.mlflow_stub import log_run


class TestLogRun:
    def test_returns_string(self) -> None:
        result = log_run({"mae": 0.1, "rmse": 0.2})
        assert isinstance(result, str)

    def test_returns_local_stub_without_mlflow(self) -> None:
        with patch.dict("sys.modules", {"mlflow": None}):
            result = log_run({"mae": 0.1})
        assert isinstance(result, str)

    def test_accepts_params(self) -> None:
        result = log_run({"mae": 0.1}, params={"n_estimators": 100})
        assert isinstance(result, str)

    def test_accepts_tags(self) -> None:
        result = log_run({"mae": 0.1}, tags={"env": "test"})
        assert isinstance(result, str)

    def test_metrics_not_empty(self) -> None:
        result = log_run({"accuracy": 0.95, "f1": 0.93})
        assert len(result) > 0

    @pytest.mark.parametrize(
        "metrics",
        [
            {"mae": 0.1},
            {"r2": 0.95, "rmse": 0.2},
            {"accuracy": 0.8, "precision": 0.9, "recall": 0.85},
        ],
    )
    def test_various_metric_dicts(self, metrics: dict) -> None:
        result = log_run(metrics)
        assert isinstance(result, str)
