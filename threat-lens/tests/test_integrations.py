"""Tests for experiment tracking and artefact storage."""

import json

import pytest

from app.aws_stub import ModelArtifactStore
from app.experiment_tracker import ExperimentTracker

# ── Experiment tracker ────────────────────────────────────────────────────────


def test_falls_back_to_local_without_mlflow_uri(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr("app.experiment_tracker.LOCAL_RUNS_PATH", path)

    tracker = ExperimentTracker()
    assert tracker.enabled is False
    assert tracker.log_run({"n_estimators": 150}, {"accuracy": 0.98}) is False
    assert path.exists()


def test_local_run_roundtrip(tmp_path, monkeypatch) -> None:
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr("app.experiment_tracker.LOCAL_RUNS_PATH", path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    tracker = ExperimentTracker(experiment_name="unit-test")
    tracker.log_run({"depth": 5}, {"accuracy": 0.9}, tags={"stage": "test"})

    runs = ExperimentTracker.read_local_runs(path)
    assert len(runs) == 1
    assert runs[0]["experiment"] == "unit-test"
    assert runs[0]["metrics"]["accuracy"] == 0.9
    assert runs[0]["tags"]["stage"] == "test"


def test_read_local_runs_missing_file(tmp_path) -> None:
    assert ExperimentTracker.read_local_runs(tmp_path / "absent.jsonl") == []


def test_read_local_runs_skips_malformed(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps({"experiment": "ok"}) + "\nNOT JSON\n\n")
    runs = ExperimentTracker.read_local_runs(path)
    assert len(runs) == 1


def test_multiple_runs_append(tmp_path, monkeypatch) -> None:
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr("app.experiment_tracker.LOCAL_RUNS_PATH", path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    tracker = ExperimentTracker()
    for i in range(3):
        tracker.log_run({"seed": i}, {"accuracy": 0.9 + i / 100})
    assert len(ExperimentTracker.read_local_runs(path)) == 3


# ── Artefact store ────────────────────────────────────────────────────────────


def test_store_falls_back_to_memory(monkeypatch) -> None:
    monkeypatch.delenv("S3_BUCKET", raising=False)
    assert ModelArtifactStore().enabled is False


def test_upload_download_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("S3_BUCKET", raising=False)
    src = tmp_path / "model.joblib"
    src.write_bytes(b"fake-model-bytes")

    store = ModelArtifactStore()
    uri = store.upload(src)
    assert uri.startswith("memory://")

    dest = tmp_path / "restored.joblib"
    assert store.download("model.joblib", dest).read_bytes() == b"fake-model-bytes"


def test_upload_missing_file_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("S3_BUCKET", raising=False)
    with pytest.raises(FileNotFoundError):
        ModelArtifactStore().upload(tmp_path / "nope.joblib")


def test_download_unknown_key_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("S3_BUCKET", raising=False)
    with pytest.raises(KeyError):
        ModelArtifactStore().download("absent.joblib", tmp_path / "out")


def test_exists_reflects_uploads(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("S3_BUCKET", raising=False)
    store = ModelArtifactStore()
    assert store.exists("model.joblib") is False

    src = tmp_path / "model.joblib"
    src.write_bytes(b"x")
    store.upload(src)
    assert store.exists("model.joblib") is True


def test_list_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("S3_BUCKET", raising=False)
    store = ModelArtifactStore()
    for name in ("a.bin", "b.bin"):
        p = tmp_path / name
        p.write_bytes(b"data")
        store.upload(p)
    assert len(store.list_artifacts()) == 2


def test_custom_prefix_used_in_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("S3_BUCKET", raising=False)
    store = ModelArtifactStore(prefix="custom")
    src = tmp_path / "m.bin"
    src.write_bytes(b"x")
    assert store.upload(src) == "memory://custom/m.bin"
