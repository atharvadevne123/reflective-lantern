"""Tests for trade-sage app/mlflow_stub.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.mlflow_stub import log_run


class TestLogRun:
    def test_returns_none_when_mlflow_not_configured(self) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", False):
            result = log_run("test-run", {"mae": 0.1})
        assert result is None

    def test_accepts_params(self) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", False):
            result = log_run("run1", {"rmse": 0.2}, params={"lr": 0.01})
        assert result is None

    def test_does_not_raise(self) -> None:
        log_run("run2", {"accuracy": 0.95})

    @pytest.mark.parametrize(
        "run_name,metrics",
        [
            ("run-a", {"mae": 0.1}),
            ("run-b", {"r2": 0.9, "rmse": 0.15}),
            ("run-c", {"auc": 0.85}),
        ],
    )
    def test_various_metrics(self, run_name: str, metrics: dict) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", False):
            log_run(run_name, metrics)
