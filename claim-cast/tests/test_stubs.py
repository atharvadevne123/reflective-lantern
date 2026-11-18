"""Tests for MLflow and S3 stubs."""
from __future__ import annotations

import pytest

from app.mlflow_stub import end_run, get_runs, log_metric, log_param, start_run
from app.s3_stub import download_artifact, list_artifacts, upload_artifact


def test_mlflow_run_lifecycle():
    run = start_run("test-run")
    assert run["run_id"].startswith("stub_")
    assert run["status"] == "RUNNING"
    log_param(run, "n_estimators", 100)
    log_metric(run, "f1", 0.85)
    end_run(run)
    assert run["status"] == "FINISHED"
    assert run["params"]["n_estimators"] == "100"
    assert run["metrics"]["f1"] == 0.85


def test_mlflow_get_runs():
    start_run("run-a")
    runs = get_runs()
    assert len(runs) >= 1


def test_s3_upload_returns_uri():
    uri = upload_artifact("/tmp/dummy.joblib", "models/dummy.joblib")
    assert uri.startswith("s3://")
    assert "dummy.joblib" in uri


def test_s3_download_validates_uri():
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        download_artifact("not-s3://bucket/key", "/tmp/out.joblib")


def test_s3_list_artifacts_returns_list():
    result = list_artifacts("models")
    assert isinstance(result, list)
