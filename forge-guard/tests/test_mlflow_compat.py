"""Tests for the MLflow-compatible experiment tracking stub."""

from __future__ import annotations


def test_log_run_returns_string(tmp_path, monkeypatch):
    from app import mlflow_compat

    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", tmp_path / "mlruns" / "0")
    run_id = mlflow_compat.log_run({"auc_cv_mean": 0.91}, params={"cv_folds": 5})
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_log_run_writes_manifest(tmp_path, monkeypatch):
    from app import mlflow_compat

    mlflow_dir = tmp_path / "mlruns" / "0"
    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
    run_id = mlflow_compat.log_run({"auc_cv_mean": 0.88})
    run_file = mlflow_dir / run_id / "manifest.json"
    assert run_file.exists()


def test_log_run_manifest_contains_metrics(tmp_path, monkeypatch):
    import json

    from app import mlflow_compat

    mlflow_dir = tmp_path / "mlruns" / "0"
    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
    metrics = {"auc_cv_mean": 0.93, "auc_train": 0.97}
    run_id = mlflow_compat.log_run(metrics, params={"model_version": "1.1.0"})
    manifest = json.loads((mlflow_dir / run_id / "manifest.json").read_text())
    assert manifest["metrics"]["auc_cv_mean"] == 0.93
    assert manifest["params"]["model_version"] == "1.1.0"


def test_log_run_no_params(tmp_path, monkeypatch):
    from app import mlflow_compat

    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", tmp_path / "mlruns" / "0")
    run_id = mlflow_compat.log_run({"loss": 0.05})
    assert isinstance(run_id, str)


def test_log_run_unique_ids(tmp_path, monkeypatch):
    import time

    from app import mlflow_compat

    monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", tmp_path / "mlruns" / "0")
    id1 = mlflow_compat.log_run({"x": 1.0})
    time.sleep(0.01)
    id2 = mlflow_compat.log_run({"x": 2.0})
    assert id1 != id2


class TestMlflowCompatEdgeCases:
    """Edge-case tests for the MLflow-compatible experiment tracking stub."""

    def test_empty_metrics_dict(self, tmp_path, monkeypatch):
        from app import mlflow_compat

        monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", tmp_path / "mlruns" / "0")
        run_id = mlflow_compat.log_run({})
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_manifest_run_id_matches_returned(self, tmp_path, monkeypatch):
        import json

        from app import mlflow_compat

        mlflow_dir = tmp_path / "mlruns" / "0"
        monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
        run_id = mlflow_compat.log_run({"auc": 0.9})
        manifest = json.loads((mlflow_dir / run_id / "manifest.json").read_text())
        assert manifest["run_id"] == run_id

    def test_empty_params_stored_as_empty_dict(self, tmp_path, monkeypatch):
        import json

        from app import mlflow_compat

        mlflow_dir = tmp_path / "mlruns" / "0"
        monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
        run_id = mlflow_compat.log_run({"loss": 0.1})
        manifest = json.loads((mlflow_dir / run_id / "manifest.json").read_text())
        assert manifest["params"] == {}

    def test_multiple_metrics_all_stored(self, tmp_path, monkeypatch):
        import json

        from app import mlflow_compat

        mlflow_dir = tmp_path / "mlruns" / "0"
        monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
        metrics = {"a": 0.1, "b": 0.2, "c": 0.3}
        run_id = mlflow_compat.log_run(metrics)
        manifest = json.loads((mlflow_dir / run_id / "manifest.json").read_text())
        for k, v in metrics.items():
            assert manifest["metrics"][k] == v

    def test_multiple_params_all_stored(self, tmp_path, monkeypatch):
        import json

        from app import mlflow_compat

        mlflow_dir = tmp_path / "mlruns" / "0"
        monkeypatch.setattr(mlflow_compat, "MLFLOW_DIR", mlflow_dir)
        params = {"lr": "0.01", "epochs": "100", "model": "xgb"}
        run_id = mlflow_compat.log_run({"loss": 0.05}, params=params)
        manifest = json.loads((mlflow_dir / run_id / "manifest.json").read_text())
        for k, v in params.items():
            assert manifest["params"][k] == v
