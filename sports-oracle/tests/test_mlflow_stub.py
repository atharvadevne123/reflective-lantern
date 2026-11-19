"""Tests for Sports-Oracle MLflow stub."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.mlflow_stub as stub_module


@pytest.fixture(autouse=True)
def temp_mlflow_path(tmp_path, monkeypatch):
    runs_path = tmp_path / "mlflow_runs.jsonl"
    monkeypatch.setattr(stub_module, "MLFLOW_RUNS_PATH", runs_path)
    yield runs_path


def test_log_run_returns_string(temp_mlflow_path):
    result = stub_module.log_run("test_run", {"accuracy": 0.9})
    assert isinstance(result, str)
    assert len(result) > 0


def test_log_run_creates_file(temp_mlflow_path):
    stub_module.log_run("test_run", {"roc_auc": 0.95})
    assert temp_mlflow_path.exists()


def test_log_run_entry_has_metrics(temp_mlflow_path):
    stub_module.log_run("my_run", {"f1": 0.88}, params={"n": 100})
    lines = [line for line in temp_mlflow_path.read_text().strip().split("\n") if line]
    entry = json.loads(lines[-1])
    assert entry["metrics"]["f1"] == 0.88


def test_log_run_entry_has_run_name(temp_mlflow_path):
    stub_module.log_run("named_run", {})
    lines = [line for line in temp_mlflow_path.read_text().strip().split("\n") if line]
    entry = json.loads(lines[-1])
    assert entry["run_name"] == "named_run"


def test_log_run_with_tags(temp_mlflow_path):
    stub_module.log_run("tagged", {}, tags={"env": "prod"})
    lines = [line for line in temp_mlflow_path.read_text().strip().split("\n") if line]
    entry = json.loads(lines[-1])
    assert entry["tags"].get("env") == "prod"


def test_multiple_runs_appended(temp_mlflow_path):
    stub_module.log_run("run1", {"acc": 0.9})
    stub_module.log_run("run2", {"acc": 0.95})
    lines = [line for line in temp_mlflow_path.read_text().strip().split("\n") if line]
    assert len(lines) == 2


def test_run_id_unique(temp_mlflow_path):
    id1 = stub_module.log_run("r1", {})
    id2 = stub_module.log_run("r2", {})
    assert id1 != id2
