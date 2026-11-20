"""Retraining pipeline tests."""
from __future__ import annotations

import os

os.environ.setdefault("MODEL_DIR", "./test_models")


def test_pipeline_runs_without_error():
    from pipelines.retrain_dag import run_pipeline
    result = run_pipeline()
    assert isinstance(result, dict)


def test_task_check_drift_returns_dict():
    from pipelines.retrain_dag import task_check_drift
    result = task_check_drift()
    assert "should_retrain" in result
    assert "breach_auc_mean" in result
    assert isinstance(result["should_retrain"], bool)


def test_task_generate_data_returns_count():
    from pipelines.retrain_dag import task_generate_training_data
    count = task_generate_training_data()
    assert count > 0
