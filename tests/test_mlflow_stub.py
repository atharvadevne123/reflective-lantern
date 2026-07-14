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
