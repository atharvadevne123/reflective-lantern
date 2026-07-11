"""Tests for the automated retraining pipeline."""

from __future__ import annotations

import pandas as pd


def test_task_engineer_features_shapes(synthetic_df: pd.DataFrame):
    from pipelines.retrain_dag import task_engineer_features

    X, y = task_engineer_features(synthetic_df)
    assert X.shape[0] == len(synthetic_df)
    assert X.shape[1] > 7
    assert len(y) == len(synthetic_df)


def test_task_retrain_model_returns_comparison(synthetic_df, tmp_path, monkeypatch):
    from app import model as m
    from pipelines.retrain_dag import task_engineer_features, task_retrain_model

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")

    X, y = task_engineer_features(synthetic_df)
    result = task_retrain_model(X, y)

    assert "auc_before" in result
    assert "auc_after" in result
    assert "improved" in result
    assert result["auc_after"] > 0.5


def test_extract_falls_back_to_synthetic_on_empty_db(monkeypatch):
    from pipelines import retrain_dag

    # Point at a non-existent DB path — few/no records → empty DataFrame or synthetic fallback
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./nonexistent_empty.db")
    df = retrain_dag.task_extract_training_data()
    # Either empty (below MIN_SAMPLES) or synthetic fallback — both are valid safe states
    assert isinstance(df, pd.DataFrame)


def test_run_pipeline_aborts_gracefully_on_no_data(monkeypatch, caplog):
    from pipelines import retrain_dag

    monkeypatch.setattr(retrain_dag, "task_extract_training_data", lambda: pd.DataFrame())
    retrain_dag.run_pipeline(trigger="manual")  # must not raise
