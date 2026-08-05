"""Tests for the champion/challenger retraining pipeline."""

from __future__ import annotations

import json

import pytest

from pipelines.retrain_dag import (
    _fetch_training_data,
    drift_report_task,
    retrain_task,
)


class TestFetchTrainingData:
    def test_returns_features_and_labels(self):
        X, y = _fetch_training_data()
        assert len(X) == len(y)

    def test_labels_are_binary(self):
        _, y = _fetch_training_data()
        assert set(y.unique()) <= {0, 1}

    def test_feature_columns_present(self):
        from app.features import INTERACTION_COLS, ITEM_COLS, USER_COLS

        X, _ = _fetch_training_data()
        for col in USER_COLS + ITEM_COLS + INTERACTION_COLS:
            assert col in X.columns


class TestRetrainTask:
    def test_skips_when_below_row_gate(self, monkeypatch):
        """A thin data window must skip rather than train on noise."""
        monkeypatch.setattr("pipelines.retrain_dag.MIN_ROWS", 10_000_000)
        result = retrain_task()
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_data"

    def test_promotes_when_auc_clears_gates(self, monkeypatch, tmp_path):
        monkeypatch.setattr("pipelines.retrain_dag.MIN_ROWS", 10)
        monkeypatch.setattr("pipelines.retrain_dag.TRAIN_ROWS", 120)
        monkeypatch.setattr("pipelines.retrain_dag.AUC_GATE", 0.0)
        monkeypatch.setattr("app.model.MODEL_PATH", tmp_path / "m.joblib")
        monkeypatch.setattr("app.model.METRICS_PATH", tmp_path / "metrics.json")
        result = retrain_task()
        assert result["status"] == "promoted"
        assert 0.0 <= result["auc"] <= 1.0

    def test_rejects_when_below_auc_gate(self, monkeypatch, tmp_path):
        """An unreachable AUC gate must block promotion, leaving the champion in place."""
        monkeypatch.setattr("pipelines.retrain_dag.MIN_ROWS", 10)
        monkeypatch.setattr("pipelines.retrain_dag.TRAIN_ROWS", 120)
        monkeypatch.setattr("pipelines.retrain_dag.AUC_GATE", 1.01)
        monkeypatch.setattr("app.model.MODEL_PATH", tmp_path / "m.joblib")
        monkeypatch.setattr("app.model.METRICS_PATH", tmp_path / "metrics.json")
        result = retrain_task()
        assert result["status"] == "rejected"

    def test_rejects_challenger_worse_than_champion(self, monkeypatch, tmp_path):
        """A challenger must not replace a stronger champion."""
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"auc_mean": 0.999}))
        monkeypatch.setattr("pipelines.retrain_dag.MIN_ROWS", 10)
        monkeypatch.setattr("pipelines.retrain_dag.TRAIN_ROWS", 120)
        monkeypatch.setattr("pipelines.retrain_dag.AUC_GATE", 0.0)
        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", metrics_path)
        monkeypatch.setattr("app.model.MODEL_PATH", tmp_path / "m.joblib")
        monkeypatch.setattr("app.model.METRICS_PATH", tmp_path / "challenger.json")
        result = retrain_task()
        assert result["status"] == "rejected"
        assert result["champion_auc"] == pytest.approx(0.999)

    def test_corrupt_champion_metrics_do_not_crash(self, monkeypatch, tmp_path):
        """Unreadable champion metrics must degrade to 'no champion', not raise."""
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text("{not valid json")
        monkeypatch.setattr("pipelines.retrain_dag.MIN_ROWS", 10)
        monkeypatch.setattr("pipelines.retrain_dag.TRAIN_ROWS", 120)
        monkeypatch.setattr("pipelines.retrain_dag.AUC_GATE", 0.0)
        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", metrics_path)
        monkeypatch.setattr("app.model.MODEL_PATH", tmp_path / "m.joblib")
        monkeypatch.setattr("app.model.METRICS_PATH", tmp_path / "challenger.json")
        result = retrain_task()
        assert result["status"] == "promoted"
        assert result["champion_auc"] == 0.0


class TestDriftReportTask:
    def test_returns_drift_summary(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = drift_report_task()
        assert "drifted" in result
        assert result["total_checked"] == 3

    def test_writes_report_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        drift_report_task()
        reports = list((tmp_path / "history" / "reports").glob("cart_mind_drift_*.json"))
        assert len(reports) == 1

    def test_report_is_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        drift_report_task()
        report = next((tmp_path / "history" / "reports").glob("cart_mind_drift_*.json"))
        data = json.loads(report.read_text())
        assert "drifted" in data
        assert "results" in data


class TestDagModuleContract:
    def test_module_imports_without_airflow(self):
        """The DAG module must import cleanly on a host with no Airflow installed."""
        import pipelines.retrain_dag as dag_module

        assert callable(dag_module.retrain_task)
        assert callable(dag_module.drift_report_task)

    def test_dag_id_is_stable(self):
        from pipelines.retrain_dag import DAG_ID

        assert DAG_ID == "cart_mind_weekly_retrain"


class TestChampionComparisonRegression:
    """Regression cover for the champion-compared-against-itself bug.

    ``train_model`` writes ``metrics.json`` as a side effect. Reading the champion
    after training therefore loaded the challenger's own metrics, so every
    comparison was ``auc >= auc`` — always true. Every challenger was promoted,
    including outright regressions, and the gate reported the challenger's AUC
    back as the champion's.
    """

    def test_champion_auc_is_not_the_challenger_auc(self, monkeypatch, tmp_path):
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"auc_mean": 0.95}))
        monkeypatch.setattr("pipelines.retrain_dag.MIN_ROWS", 10)
        monkeypatch.setattr("pipelines.retrain_dag.TRAIN_ROWS", 120)
        monkeypatch.setattr("pipelines.retrain_dag.AUC_GATE", 0.0)
        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", metrics_path)
        monkeypatch.setattr("app.model.MODEL_PATH", tmp_path / "m.joblib")
        monkeypatch.setattr("app.model.METRICS_PATH", tmp_path / "challenger.json")

        result = retrain_task()
        assert result["champion_auc"] != result["auc"]
        assert result["champion_auc"] == pytest.approx(0.95)

    def test_weak_challenger_cannot_displace_strong_champion(self, monkeypatch, tmp_path):
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"auc_mean": 0.99}))
        monkeypatch.setattr("pipelines.retrain_dag.MIN_ROWS", 10)
        monkeypatch.setattr("pipelines.retrain_dag.TRAIN_ROWS", 120)
        monkeypatch.setattr("pipelines.retrain_dag.AUC_GATE", 0.0)
        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", metrics_path)
        monkeypatch.setattr("app.model.MODEL_PATH", tmp_path / "m.joblib")
        monkeypatch.setattr("app.model.METRICS_PATH", tmp_path / "challenger.json")

        result = retrain_task()
        assert result["status"] == "rejected"

    def test_rejected_run_preserves_champion_metrics(self, monkeypatch, tmp_path):
        """A rejected challenger must leave the champion's metrics intact on disk."""
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"auc_mean": 0.99}))
        monkeypatch.setattr("pipelines.retrain_dag.MIN_ROWS", 10)
        monkeypatch.setattr("pipelines.retrain_dag.TRAIN_ROWS", 120)
        monkeypatch.setattr("pipelines.retrain_dag.AUC_GATE", 0.0)
        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", metrics_path)
        monkeypatch.setattr("app.model.MODEL_PATH", tmp_path / "m.joblib")
        monkeypatch.setattr("app.model.METRICS_PATH", tmp_path / "challenger.json")

        retrain_task()
        assert json.loads(metrics_path.read_text())["auc_mean"] == pytest.approx(0.99)


class TestReadChampionAuc:
    def test_missing_file_returns_zero(self, monkeypatch, tmp_path):
        from pipelines.retrain_dag import read_champion_auc

        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", tmp_path / "absent.json")
        assert read_champion_auc() == 0.0

    def test_valid_file_returns_auc(self, monkeypatch, tmp_path):
        from pipelines.retrain_dag import read_champion_auc

        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"auc_mean": 0.812}))
        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", path)
        assert read_champion_auc() == pytest.approx(0.812)

    def test_corrupt_file_returns_zero(self, monkeypatch, tmp_path):
        from pipelines.retrain_dag import read_champion_auc

        path = tmp_path / "metrics.json"
        path.write_text("{ not json")
        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", path)
        assert read_champion_auc() == 0.0

    def test_missing_key_returns_zero(self, monkeypatch, tmp_path):
        from pipelines.retrain_dag import read_champion_auc

        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"other_key": 1}))
        monkeypatch.setattr("pipelines.retrain_dag.METRICS_PATH", path)
        assert read_champion_auc() == 0.0
