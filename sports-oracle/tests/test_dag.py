"""Airflow DAG function tests for Sports-Oracle."""

from __future__ import annotations

import pytest


def test_collect_training_data_returns_valid_stats():
    """collect_training_data should return n_samples >= MIN_SAMPLES."""
    from pipelines.retrain_dag import collect_training_data

    class FakeTaskInstance:
        def xcom_push(self, key, value):
            pass

    result = collect_training_data(ti=FakeTaskInstance())
    assert result["n_samples"] >= 500


def test_drift_check_runs_without_error():
    """drift_check should run without raising."""
    from pipelines.retrain_dag import drift_check

    class FakeTaskInstance:
        def xcom_push(self, key, value):
            pass

    drift_check(ti=FakeTaskInstance())


def test_promote_champion_rejects_bad_challenger(tmp_path):
    """Challenger with AUC below MIN_AUC should not be promoted."""
    import shutil
    from pathlib import Path

    from pipelines.retrain_dag import promote_champion

    # Create a fake challenger.joblib to avoid FileNotFoundError
    challenger_path = Path("challenger.joblib")
    champion_path = Path("model.joblib")
    challenger_path.write_bytes(b"fake")
    champion_path.write_bytes(b"champion")

    class FakeTaskInstance:
        def xcom_pull(self, key, task_ids):
            if key == "challenger_auc":
                return 0.3  # Below MIN_AUC of 0.65
            if key == "champion_auc":
                return 0.7
            return None

    try:
        promote_champion(ti=FakeTaskInstance())
        # Champion should not have been overwritten since challenger_auc < MIN_AUC
        assert champion_path.read_bytes() == b"champion"
    finally:
        for p in (challenger_path, champion_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
