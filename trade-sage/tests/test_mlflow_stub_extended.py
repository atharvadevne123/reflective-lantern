"""Extended tests for mlflow_stub safe-logging utilities."""

from __future__ import annotations

from unittest.mock import patch

from app.mlflow_stub import is_mlflow_available, log_run_safe


class TestIsMlflowAvailable:
    def test_returns_bool(self) -> None:
        result = is_mlflow_available()
        assert isinstance(result, bool)

    def test_false_when_mlflow_not_configured(self) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", False):
            assert is_mlflow_available() is False

    def test_true_when_mlflow_configured(self) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", True):
            assert is_mlflow_available() is True


class TestLogRunSafe:
    def test_returns_false_when_not_available(self) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", False):
            result = log_run_safe("test_run", {"auc": 0.9})
            assert result is False

    def test_returns_bool(self) -> None:
        result = log_run_safe("test_run", {"auc": 0.85})
        assert isinstance(result, bool)

    def test_never_raises_on_error(self) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", True):
            with patch("app.mlflow_stub.log_run", side_effect=RuntimeError("connection error")):
                result = log_run_safe("run", {"metric": 1.0})
                assert result is False

    def test_accepts_params_kwarg(self) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", False):
            result = log_run_safe("run", {"auc": 0.9}, params={"lr": 0.01})
            assert result is False

    def test_empty_metrics_does_not_raise(self) -> None:
        with patch("app.mlflow_stub._MLFLOW_AVAILABLE", False):
            result = log_run_safe("run", {})
            assert result is False
