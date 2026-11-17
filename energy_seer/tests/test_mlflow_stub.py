"""Tests for Energy-Seer MLflow stub."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mlflow_stub import end_run, log_metrics, log_model, log_params, start_run


def test_log_params_does_not_raise():
    log_params({"lr": 0.01, "n_estimators": 100})


def test_log_metrics_does_not_raise():
    log_metrics({"accuracy": 0.95, "roc_auc": 0.98})


def test_log_metrics_with_step():
    log_metrics({"loss": 0.1}, step=5)


def test_log_model_does_not_raise():
    log_model("/tmp/model.joblib", artifact_name="model")


def test_start_run_does_not_raise():
    start_run(run_name="test-run")


def test_end_run_does_not_raise():
    end_run()


def test_start_end_run_cycle():
    start_run("cycle-test")
    end_run()


def test_log_params_empty_dict():
    log_params({})


def test_log_metrics_empty_dict():
    log_metrics({})


@pytest.mark.parametrize("run_name", ["run-a", "run-b", "energy-seer"])
def test_start_run_with_various_names(run_name):
    start_run(run_name=run_name)
    end_run()
