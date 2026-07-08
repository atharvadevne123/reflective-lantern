"""Tests for the automated retraining pipeline."""

from __future__ import annotations

from pathlib import Path

from pipelines.retrain_dag import (
    task_check_drift,
    task_promote_model,
    task_validate_model,
)


def test_check_drift_returns_expected_keys() -> None:
    result = task_check_drift(db_url="sqlite:///./test_tp.db")
    assert "drift_count" in result
    assert "needs_retrain" in result


def test_check_drift_missing_table_is_safe() -> None:
    result = task_check_drift(db_url="sqlite:///./nonexistent_tp.db")
    assert result["drift_count"] == 0
    assert result["needs_retrain"] is False
    Path("nonexistent_tp.db").unlink(missing_ok=True)


def test_validate_model_missing_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = task_validate_model()
    assert result["passed"] is False


def test_validate_model_passes_above_threshold(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "metrics.json").write_text('{"ensemble_auc": 0.93}')
    result = task_validate_model(auc_threshold=0.75)
    assert result["passed"] is True
    assert result["ensemble_auc"] == 0.93


def test_validate_model_fails_below_threshold(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "metrics.json").write_text('{"ensemble_auc": 0.60}')
    result = task_validate_model(auc_threshold=0.75)
    assert result["passed"] is False


def test_promote_skips_on_failed_validation() -> None:
    result = task_promote_model({"passed": False})
    assert result["status"] == "skipped"


def test_promote_copies_model(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"fake-model-bytes")
    result = task_promote_model({"passed": True})
    assert result["status"] == "promoted"
    assert (tmp_path / "model_stable.joblib").exists()
