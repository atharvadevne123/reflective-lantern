"""MLflow stub tests."""

from __future__ import annotations

import pytest

from app.mlflow_stub import get_best_run, log_metrics


def test_log_and_retrieve(tmp_path, monkeypatch):
    import app.mlflow_stub as ms

    monkeypatch.setattr(ms, "_RUN_LOG", tmp_path / "runs.jsonl")
    log_metrics("run-1", {"r2_mean": 0.85, "mae_kwh": 1.2})
    log_metrics("run-2", {"r2_mean": 0.92, "mae_kwh": 0.9})
    best = get_best_run("r2_mean")
    assert best is not None
    assert best["metrics"]["r2_mean"] == pytest.approx(0.92)


def test_best_run_no_log(tmp_path, monkeypatch):
    import app.mlflow_stub as ms

    monkeypatch.setattr(ms, "_RUN_LOG", tmp_path / "missing.jsonl")
    assert get_best_run() is None


def test_log_returns_run_name(tmp_path, monkeypatch):
    import app.mlflow_stub as ms

    monkeypatch.setattr(ms, "_RUN_LOG", tmp_path / "runs.jsonl")
    result = log_metrics("my-run", {"r2_mean": 0.7})
    assert result == "my-run"
