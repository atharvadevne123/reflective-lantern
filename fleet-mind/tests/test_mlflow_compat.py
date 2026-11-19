"""Tests for Fleet-Mind MLflow compatibility stub."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def use_temp_mlflow_log(tmp_path, monkeypatch):
    log_path = tmp_path / "mlflow.jsonl"
    monkeypatch.setenv("MLFLOW_FALLBACK_LOG", str(log_path))
    import app.mlflow_compat as mc

    mc._FALLBACK_LOG = log_path
    yield log_path


def test_log_training_run_returns_string(use_temp_mlflow_log):
    import app.mlflow_compat as mc

    result = mc.log_training_run("test_run", {"lr": 0.01}, {"accuracy": 0.95})
    assert isinstance(result, str)
    assert len(result) > 0


def test_log_training_run_creates_log_file(use_temp_mlflow_log):
    import app.mlflow_compat as mc

    mc.log_training_run("run1", {}, {"roc_auc": 0.9})
    assert use_temp_mlflow_log.exists()


def test_log_training_run_entry_has_params_and_metrics(use_temp_mlflow_log):
    import app.mlflow_compat as mc

    mc.log_training_run("run1", {"n_estimators": 100}, {"f1": 0.88})
    lines = [line for line in use_temp_mlflow_log.read_text().strip().split("\n") if line]
    entry = json.loads(lines[-1])
    assert entry["run_name"] == "run1"
    assert entry["params"]["n_estimators"] == 100
    assert entry["metrics"]["f1"] == 0.88


def test_log_training_run_with_tags(use_temp_mlflow_log):
    import app.mlflow_compat as mc

    mc.log_training_run("tagged", {}, {}, tags={"env": "test"})
    lines = [line for line in use_temp_mlflow_log.read_text().strip().split("\n") if line]
    entry = json.loads(lines[-1])
    assert entry.get("tags", {}).get("env") == "test"


def test_multiple_runs_appended(use_temp_mlflow_log):
    import app.mlflow_compat as mc

    mc.log_training_run("run1", {}, {"acc": 0.9})
    mc.log_training_run("run2", {}, {"acc": 0.95})
    lines = [line for line in use_temp_mlflow_log.read_text().strip().split("\n") if line]
    assert len(lines) == 2


def test_log_training_run_entry_has_timestamp(use_temp_mlflow_log):
    import app.mlflow_compat as mc

    mc.log_training_run("timed", {}, {})
    lines = [line for line in use_temp_mlflow_log.read_text().strip().split("\n") if line]
    entry = json.loads(lines[-1])
    assert "logged_at" in entry
