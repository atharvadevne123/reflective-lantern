"""Tests for MLflow stub."""

from __future__ import annotations

import pytest

from app.mlflow_stub import dump_run_store, get_best_run, log_metrics


class TestMlflowStub:
    def setup_method(self):
        from app import mlflow_stub
        mlflow_stub._RUN_STORE.clear()

    def test_log_metrics_returns_run_id(self):
        run_id = log_metrics({"r2_mean": 0.85, "rmse_mean": 250.0})
        assert run_id.startswith("run_")

    def test_multiple_runs_increment(self):
        id1 = log_metrics({"r2_mean": 0.80})
        id2 = log_metrics({"r2_mean": 0.85})
        assert id1 != id2

    def test_get_best_run_by_metric(self):
        log_metrics({"r2_mean": 0.80}, run_name="run_a")
        log_metrics({"r2_mean": 0.90}, run_name="run_b")
        log_metrics({"r2_mean": 0.75}, run_name="run_c")
        best = get_best_run("r2_mean")
        assert best is not None
        assert best["r2_mean"] == pytest.approx(0.90, rel=0.01)
        assert best["run_name"] == "run_b"

    def test_get_best_run_no_runs(self):
        assert get_best_run("r2_mean") is None

    def test_dump_run_store(self, tmp_path):
        log_metrics({"r2_mean": 0.85})
        path = str(tmp_path / "runs.json")
        dump_run_store(path)
        import json
        data = json.loads(open(path).read())
        assert len(data) == 1
        assert data[0]["r2_mean"] == pytest.approx(0.85)
