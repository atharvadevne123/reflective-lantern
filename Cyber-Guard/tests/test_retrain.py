"""Retraining pipeline tests.

The accuracy gate is the safety interlock on automated retraining: if it
fails open, a model trained on a bad week silently replaces a good one.
"""

from __future__ import annotations

import json
import os

import pytest

from pipelines import retrain_dag


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run inside a temp directory so retrain_metrics.json is isolated."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_evaluate_passes_above_floor(in_tmp_cwd, monkeypatch):
    monkeypatch.setenv("RETRAIN_ACCURACY_FLOOR", "0.70")
    (in_tmp_cwd / "retrain_metrics.json").write_text(json.dumps({"accuracy_mean": 0.95}))
    retrain_dag._evaluate_model()  # must not raise


def test_evaluate_rejects_below_floor(in_tmp_cwd, monkeypatch):
    """A degraded model must abort the DAG rather than be promoted."""
    monkeypatch.setenv("RETRAIN_ACCURACY_FLOOR", "0.70")
    (in_tmp_cwd / "retrain_metrics.json").write_text(json.dumps({"accuracy_mean": 0.42}))
    with pytest.raises(ValueError, match="below the"):
        retrain_dag._evaluate_model()


def test_evaluate_respects_configured_floor(in_tmp_cwd, monkeypatch):
    """The floor is configurable, not hard-coded at 0.70."""
    monkeypatch.setenv("RETRAIN_ACCURACY_FLOOR", "0.99")
    (in_tmp_cwd / "retrain_metrics.json").write_text(json.dumps({"accuracy_mean": 0.95}))
    with pytest.raises(ValueError):
        retrain_dag._evaluate_model()


def test_evaluate_raises_when_metrics_missing(in_tmp_cwd):
    with pytest.raises(FileNotFoundError):
        retrain_dag._evaluate_model()


def test_evaluate_treats_absent_accuracy_as_zero(in_tmp_cwd, monkeypatch):
    """A metrics file without accuracy must fail closed, not pass."""
    monkeypatch.setenv("RETRAIN_ACCURACY_FLOOR", "0.70")
    (in_tmp_cwd / "retrain_metrics.json").write_text(json.dumps({}))
    with pytest.raises(ValueError):
        retrain_dag._evaluate_model()


def test_full_pipeline_runs(in_tmp_cwd, monkeypatch):
    """The non-Airflow entry point must complete end to end."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{in_tmp_cwd}/rt.db")
    monkeypatch.setenv("MODEL_PATH", str(in_tmp_cwd / "m.joblib"))
    monkeypatch.setenv("METRICS_PATH", str(in_tmp_cwd / "me.json"))

    from sqlalchemy import create_engine

    from app.database import Base

    Base.metadata.create_all(bind=create_engine(os.environ["DATABASE_URL"]))

    assert retrain_dag.run_retrain_pipeline() == {"status": "success"}
    assert (in_tmp_cwd / "retrain_metrics.json").exists()
