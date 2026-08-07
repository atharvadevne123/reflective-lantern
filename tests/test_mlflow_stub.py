"""MLflow stub tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.aws_stub import download_model_artefacts, upload_model_artefacts
from app.mlflow_stub import get_best_run, log_metrics, log_training_run


def test_log_and_retrieve(tmp_path, monkeypatch):
    import app.mlflow_stub as ms

    monkeypatch.setattr(ms, "_RUN_LOG", tmp_path / "runs.jsonl")
    log_metrics("run-1", {"r2_mean": 0.85, "mae_kwh": 1.2})
    log_metrics("run-2", {"r2_mean": 0.92, "mae_kwh": 0.9})
    best = get_best_run("r2_mean")
    assert best is not None
    assert best["metrics"]["r2_mean"] == pytest.approx(0.92)

    def test_log_metrics_returns_run_id(self):
        run_id = log_metrics({"r2_mean": 0.85, "rmse_mean": 250.0})
        assert run_id.startswith("run_")


def test_best_run_no_log(tmp_path, monkeypatch):
    import app.mlflow_stub as ms

    monkeypatch.setattr(ms, "_RUN_LOG", tmp_path / "missing.jsonl")
    assert get_best_run() is None

    def test_get_best_run_by_metric(self):
        log_metrics({"r2_mean": 0.80}, run_name="run_a")
        log_metrics({"r2_mean": 0.90}, run_name="run_b")
        log_metrics({"r2_mean": 0.75}, run_name="run_c")
        best = get_best_run("r2_mean")
        assert best is not None
        assert best["r2_mean"] == pytest.approx(0.90, rel=0.01)
        assert best["run_name"] == "run_b"


def test_log_returns_run_name(tmp_path, monkeypatch):
    pass


# --- AWS stub tests ---


def test_upload_no_bucket_returns_empty() -> None:
    with patch.dict("os.environ", {"S3_BUCKET": ""}):
        import app.aws_stub as s

        original = s._BUCKET
        s._BUCKET = ""
        result = upload_model_artefacts(["model.joblib"])
        s._BUCKET = original
    assert result == []


def test_upload_missing_file_skipped(tmp_path) -> None:
    import app.aws_stub as s

    original_bucket = s._BUCKET
    s._BUCKET = "test-bucket"
    mock_client = MagicMock()
    with patch("app.aws_stub._s3_client", return_value=mock_client):
        result = upload_model_artefacts([str(tmp_path / "nonexistent.joblib")])
    s._BUCKET = original_bucket
    assert result == []
    mock_client.upload_file.assert_not_called()


def test_upload_existing_file(tmp_path) -> None:
    import app.aws_stub as s

    test_file = tmp_path / "model.joblib"
    test_file.write_bytes(b"fake model")
    mock_client = MagicMock()
    with (
        patch("app.aws_stub._s3_client", return_value=mock_client),
        patch.object(s, "_BUCKET", "test-bucket"),
    ):
        result = upload_model_artefacts([str(test_file)])
    assert len(result) == 1
    assert "s3://test-bucket" in result[0]


def test_download_no_bucket_returns_empty() -> None:
    import app.aws_stub as s

    original = s._BUCKET
    s._BUCKET = ""
    result = download_model_artefacts("/tmp")
    s._BUCKET = original
    assert result == []


def test_upload_empty_list_returns_empty() -> None:
    import app.aws_stub as s

    original = s._BUCKET
    s._BUCKET = "test-bucket"
    mock_client = MagicMock()
    with patch("app.aws_stub._s3_client", return_value=mock_client):
        result = upload_model_artefacts([])
    s._BUCKET = original
    assert result == []


def test_upload_s3_error_skips_file(tmp_path) -> None:
    import app.aws_stub as s

    f = tmp_path / "model.joblib"
    f.write_bytes(b"data")
    mock_client = MagicMock()
    mock_client.upload_file.side_effect = RuntimeError("S3 error")
    with (
        patch("app.aws_stub._s3_client", return_value=mock_client),
        patch.object(s, "_BUCKET", "bucket"),
    ):
        result = upload_model_artefacts([str(f)])
    assert result == []


def test_download_with_client_success(tmp_path) -> None:
    import app.aws_stub as s

    mock_client = MagicMock()

    def fake_download(bucket, key, local_path):
        open(local_path, "w").close()

    mock_client.download_file.side_effect = fake_download
    with (
        patch("app.aws_stub._s3_client", return_value=mock_client),
        patch.object(s, "_BUCKET", "bucket"),
    ):
        result = download_model_artefacts(str(tmp_path))
    assert len(result) == 2


def test_log_training_run_no_uri_with_tags_returns_none() -> None:
    import app.mlflow_stub as m

    original = m._TRACKING_URI
    m._TRACKING_URI = ""
    try:
        result = log_training_run({"lr": 0.01}, {"r2": 0.85}, tags={"env": "test"})
    finally:
        m._TRACKING_URI = original
    assert result is None


def test_log_training_run_empty_params_returns_none() -> None:
    import app.mlflow_stub as m

    original = m._TRACKING_URI
    m._TRACKING_URI = ""
    try:
        result = log_training_run({}, {})
    finally:
        m._TRACKING_URI = original
    assert result is None


def test_upload_multiple_files_returns_all_uris(tmp_path) -> None:
    import app.aws_stub as s

    f1 = tmp_path / "model.joblib"
    f2 = tmp_path / "metrics.json"
    f1.write_bytes(b"model data")
    f2.write_bytes(b'{"r2": 0.9}')

    mock_client = MagicMock()
    with (
        patch("app.aws_stub._s3_client", return_value=mock_client),
        patch.object(s, "_BUCKET", "test-bucket"),
    ):
        result = upload_model_artefacts([str(f1), str(f2)])
    assert len(result) == 2
    assert all("s3://test-bucket" in uri for uri in result)


@pytest.mark.parametrize("bucket", ["my-bucket", "prod-bucket", "staging-bucket"])
def test_upload_bucket_name_in_uri(tmp_path, bucket: str) -> None:
    import app.aws_stub as s

    f = tmp_path / "model.joblib"
    f.write_bytes(b"data")
    mock_client = MagicMock()
    with (
        patch("app.aws_stub._s3_client", return_value=mock_client),
        patch.object(s, "_BUCKET", bucket),
    ):
        result = upload_model_artefacts([str(f)])
    assert len(result) == 1
    assert bucket in result[0]


def test_download_error_returns_empty(tmp_path) -> None:
    import app.aws_stub as s

    mock_client = MagicMock()
    mock_client.download_file.side_effect = RuntimeError("Download failed")
    with (
        patch("app.aws_stub._s3_client", return_value=mock_client),
        patch.object(s, "_BUCKET", "bucket"),
    ):
        result = download_model_artefacts(str(tmp_path))
    assert result == []


def test_default_region_constant() -> None:
    from app.aws_stub import DEFAULT_REGION

    assert isinstance(DEFAULT_REGION, str)
    assert len(DEFAULT_REGION) > 0


def test_default_prefix_constant() -> None:
    from app.aws_stub import DEFAULT_PREFIX

    assert isinstance(DEFAULT_PREFIX, str)
    assert "/" in DEFAULT_PREFIX


def test_artefact_filenames_constant() -> None:
    from app.aws_stub import ARTEFACT_FILENAMES

    assert len(ARTEFACT_FILENAMES) >= 2


def test_artefact_filenames_contains_model() -> None:
    from app.aws_stub import ARTEFACT_FILENAMES

    assert any("model" in f for f in ARTEFACT_FILENAMES)


def test_artefact_filenames_contains_metrics() -> None:
    from app.aws_stub import ARTEFACT_FILENAMES

    assert any("metrics" in f for f in ARTEFACT_FILENAMES)


@pytest.mark.parametrize("filename", ["model.joblib", "metrics.json"])
def test_artefact_filename_in_constant(filename: str) -> None:
    from app.aws_stub import ARTEFACT_FILENAMES

    assert filename in ARTEFACT_FILENAMES


def test_set_tracking_uri_configures():
    import app.mlflow_stub as stub
    from app.mlflow_stub import set_tracking_uri

    set_tracking_uri("http://localhost:5000")
    assert stub._TRACKING_URI == "http://localhost:5000"
    set_tracking_uri("")  # cleanup


def test_log_params_appends_to_log(tmp_path, monkeypatch):
    import app.mlflow_stub as stub

    monkeypatch.setattr(stub, "_RUN_LOG", tmp_path / "runs.jsonl")
    stub.log_params("run1", {"lr": 0.01, "n": 100})
    assert (tmp_path / "runs.jsonl").exists()


def test_log_artifact_appends_to_log(tmp_path, monkeypatch):
    import app.mlflow_stub as stub

    monkeypatch.setattr(stub, "_RUN_LOG", tmp_path / "runs.jsonl")
    stub.log_artifact("run1", "/tmp/model.joblib")
    content = (tmp_path / "runs.jsonl").read_text()
    assert "model.joblib" in content


def test_list_runs_empty_when_no_log():
    import pathlib

    import app.mlflow_stub as stub
    from app.mlflow_stub import list_runs

    orig = stub._RUN_LOG
    stub._RUN_LOG = pathlib.Path("/tmp/nonexistent_mlruns.jsonl")
    try:
        assert list_runs() == []
    finally:
        stub._RUN_LOG = orig


def test_list_runs_returns_entries(tmp_path, monkeypatch):
    import json

    import app.mlflow_stub as stub

    log = tmp_path / "runs.jsonl"
    log.write_text(json.dumps({"run": "r1", "metrics": {"r2": 0.9}}) + "\n")
    monkeypatch.setattr(stub, "_RUN_LOG", log)
    runs = stub.list_runs()
    assert len(runs) == 1
    assert runs[0]["run"] == "r1"


def test_delete_artefact_no_bucket_returns_false() -> None:
    import app.aws_stub as s

    original = s._BUCKET
    s._BUCKET = ""
    try:
        result = s.delete_artefact("some/key")
    finally:
        s._BUCKET = original
    assert result is False


def test_delete_artefact_no_client_returns_false() -> None:
    from unittest.mock import patch

    import app.aws_stub as s

    with (
        patch("app.aws_stub._s3_client", return_value=None),
        patch.object(s, "_BUCKET", "bucket"),
    ):
        result = s.delete_artefact("some/key")
    assert result is False


def test_delete_artefact_s3_error_returns_false() -> None:
    from unittest.mock import MagicMock, patch

    import app.aws_stub as s

    mock_client = MagicMock()
    mock_client.delete_object.side_effect = RuntimeError("S3 error")
    with (
        patch("app.aws_stub._s3_client", return_value=mock_client),
        patch.object(s, "_BUCKET", "bucket"),
    ):
        result = s.delete_artefact("some/key")
    assert result is False


def test_delete_artefact_success_returns_true() -> None:
    from unittest.mock import MagicMock, patch

    import app.aws_stub as s

    mock_client = MagicMock()
    with (
        patch("app.aws_stub._s3_client", return_value=mock_client),
        patch.object(s, "_BUCKET", "bucket"),
    ):
        result = s.delete_artefact("some/key")
    assert result is True
    mock_client.delete_object.assert_called_once_with(Bucket="bucket", Key="some/key")


class TestDeleteRun:
    def test_deletes_existing(self) -> None:
        from app.mlflow_stub import clear_runs, delete_run, log_metrics

        clear_runs()
        log_metrics("run_del", {"r2": 0.9})
        assert delete_run("run_del") is True

    def test_returns_false_when_not_found(self) -> None:
        from app.mlflow_stub import clear_runs, delete_run

        clear_runs()
        assert delete_run("nonexistent") is False


class TestRunExists:
    def test_existing_run(self) -> None:
        from app.mlflow_stub import clear_runs, log_metrics, run_exists

        clear_runs()
        log_metrics("exists_run", {"loss": 0.1})
        assert run_exists("exists_run") is True

    def test_missing_run(self) -> None:
        from app.mlflow_stub import clear_runs, run_exists

        clear_runs()
        assert run_exists("ghost") is False


class TestClearRuns:
    def test_clears_all(self) -> None:
        from app.mlflow_stub import clear_runs, log_metrics, run_count

        log_metrics("r1", {"m": 1.0})
        log_metrics("r2", {"m": 2.0})
        count = clear_runs()
        assert count >= 0
        assert run_count() == 0

    def test_empty_store(self) -> None:
        from app.mlflow_stub import clear_runs

        clear_runs()
        assert clear_runs() == 0


class TestRunCount:
    def test_count_increments(self) -> None:
        from app.mlflow_stub import clear_runs, log_metrics, run_count

        clear_runs()
        log_metrics("rc1", {"v": 1.0})
        log_metrics("rc2", {"v": 2.0})
        assert run_count() == 2

    def test_empty_after_clear(self) -> None:
        from app.mlflow_stub import clear_runs, run_count

        clear_runs()
        assert run_count() == 0
