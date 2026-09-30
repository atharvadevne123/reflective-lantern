"""MLflow tracker stub tests."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("MODEL_DIR", "./test_models")


def test_log_training_run_writes_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")

    # Override MODEL_DIR path in module
    import app.mlflow_tracker as tracker
    import importlib
    importlib.reload(tracker)

    metrics = {"breach_auc_mean": 0.82, "category_accuracy_mean": 0.75}
    result = tracker.log_training_run(metrics)
    assert result is None  # local fallback returns None


def test_log_training_run_no_crash():
    from app.mlflow_tracker import log_training_run
    metrics = {"breach_auc_mean": 0.78}
    log_training_run(metrics, params={"n_estimators": 100})
