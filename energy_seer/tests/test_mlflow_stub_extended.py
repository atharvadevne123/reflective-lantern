"""Extended tests for energy_seer app/mlflow_stub.py."""

from __future__ import annotations

import logging

import pytest

from app.mlflow_stub import end_run, log_metrics, log_model, log_params, start_run


class TestLogParamsExtended:
    def test_log_params_empty_dict(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_params({})
        assert any("mlflow_params" in r.message for r in caplog.records)

    def test_log_params_with_values(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_params({"lr": 0.01, "epochs": 10})
        assert any("mlflow_params" in r.message for r in caplog.records)

    def test_log_params_returns_none(self) -> None:
        result = log_params({"a": 1})
        assert result is None


class TestLogMetricsExtended:
    def test_log_metrics_no_step(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_metrics({"mse": 0.5})
        assert any("mlflow_metrics" in r.message for r in caplog.records)

    def test_log_metrics_with_step(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_metrics({"loss": 0.1}, step=5)
        assert any("step=5" in r.message for r in caplog.records)

    def test_log_metrics_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_metrics({})
        assert len(caplog.records) >= 1

    def test_log_metrics_returns_none(self) -> None:
        assert log_metrics({"mse": 0.1}) is None


class TestLogModelExtended:
    def test_log_model_default_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_model("/path/to/model.pkl")
        assert any("mlflow_artifact" in r.message for r in caplog.records)

    def test_log_model_custom_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_model("/path/to/model.pkl", artifact_name="energy_model")
        assert any("energy_model" in r.message for r in caplog.records)

    def test_log_model_returns_none(self) -> None:
        assert log_model("/some/path") is None


class TestStartAndEndRunExtended:
    def test_start_run_logs_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            start_run("test-run")
        assert any("test-run" in r.message for r in caplog.records)

    def test_start_run_default_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            start_run()
        assert any("mlflow_run_start" in r.message for r in caplog.records)

    def test_start_run_returns_none(self) -> None:
        assert start_run("x") is None

    def test_end_run_logs_message(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            end_run()
        assert any("mlflow_run_end" in r.message for r in caplog.records)

    def test_end_run_returns_none(self) -> None:
        assert end_run() is None


class TestLogParamsTypesExtended:
    def test_log_params_with_string_values(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_params({"model": "xgboost", "version": "1.0"})
        assert any("mlflow_params" in r.message for r in caplog.records)

    def test_log_params_with_boolean_values(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_params({"use_gpu": True, "debug": False})
        assert any("mlflow_params" in r.message for r in caplog.records)

    def test_log_metrics_multiple_metrics(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_metrics({"mse": 0.1, "mae": 0.05, "r2": 0.95})
        assert any("mlflow_metrics" in r.message for r in caplog.records)

    def test_log_model_path_in_message(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_model("/my/special/path/model.pkl")
        assert any("my/special/path" in r.message for r in caplog.records)

    def test_start_and_end_run_sequence(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            start_run("my-experiment")
            end_run()
        messages = [r.message for r in caplog.records]
        assert any("my-experiment" in m for m in messages)
        assert any("mlflow_run_end" in m for m in messages)

    @pytest.mark.parametrize("step", [0, 1, 10, 100])
    def test_log_metrics_with_various_steps(self, step: int, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.mlflow_stub"):
            log_metrics({"loss": 0.5}, step=step)
        assert any(f"step={step}" in r.message for r in caplog.records)
