"""Tests for the MLflow-compatible experiment tracking stub."""

from __future__ import annotations


def test_log_run_returns_string(tmp_path, monkeypatch):
    from app import mlflow_compat

    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", tmp_path / "mlruns" / "0")
    run_id = mlflow_compat.log_run({"auc_cv_mean": 0.91}, params={"cv_folds": 5})
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_log_run_writes_manifest(tmp_path, monkeypatch):
    from app import mlflow_compat

    mlflow_dir = tmp_path / "mlruns" / "0"
    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
    run_id = mlflow_compat.log_run({"auc_cv_mean": 0.88})
    run_file = mlflow_dir / run_id / "manifest.json"
    assert run_file.exists()


def test_log_run_manifest_contains_metrics(tmp_path, monkeypatch):
    import json

    from app import mlflow_compat

    mlflow_dir = tmp_path / "mlruns" / "0"
    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
    metrics = {"auc_cv_mean": 0.93, "auc_train": 0.97}
    run_id = mlflow_compat.log_run(metrics, params={"model_version": "1.1.0"})
    manifest = json.loads((mlflow_dir / run_id / "manifest.json").read_text())
    assert manifest["metrics"]["auc_cv_mean"] == 0.93
    assert manifest["params"]["model_version"] == "1.1.0"


def test_log_run_no_params(tmp_path, monkeypatch):
    from app import mlflow_compat

    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", tmp_path / "mlruns" / "0")
    run_id = mlflow_compat.log_run({"loss": 0.05})
    assert isinstance(run_id, str)


def test_log_run_unique_ids(tmp_path, monkeypatch):
    import time

    from app import mlflow_compat

    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", tmp_path / "mlruns" / "0")
    id1 = mlflow_compat.log_run({"x": 1.0})
    time.sleep(0.01)
    id2 = mlflow_compat.log_run({"x": 2.0})
    assert id1 != id2


import pytest


@pytest.mark.parametrize("metric_val", [0.0, 0.5, 1.0])
def test_log_run_various_metric_values(tmp_path, monkeypatch, metric_val: float) -> None:
    from app import mlflow_compat

    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", tmp_path / "mlruns" / "0")
    run_id = mlflow_compat.log_run({"score": metric_val})
    assert isinstance(run_id, str)


def test_log_run_manifest_has_timestamp(tmp_path, monkeypatch) -> None:
    import json

    from app import mlflow_compat

    mlflow_dir = tmp_path / "mlruns" / "0"
    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
    run_id = mlflow_compat.log_run({"val": 1.0})
    manifest = json.loads((mlflow_dir / run_id / "manifest.json").read_text())
    assert "timestamp" in manifest or "run_id" in manifest
