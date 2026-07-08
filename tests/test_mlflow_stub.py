"""Tests for MLflow and AWS stubs (mocked external calls)."""

from unittest.mock import MagicMock, patch

from app.aws_stub import download_model_artefacts, upload_model_artefacts
from app.mlflow_stub import log_training_run

# --- MLflow stub tests ---


def test_log_training_run_no_uri_returns_none():
    with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": ""}):
        result = log_training_run({"lr": 0.01}, {"r2": 0.85})
    assert result is None


def test_log_training_run_mlflow_not_installed_returns_none():
    with (
        patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://localhost:5000"}),
        patch("builtins.__import__", side_effect=ImportError("mlflow not installed")),
    ):
        result = log_training_run({"lr": 0.01}, {"r2": 0.85})
    assert result is None


def test_log_training_run_mlflow_exception_returns_none():
    with (
        patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://localhost:5000"}),
        patch("app.mlflow_stub.os.getenv", return_value="http://localhost:5000"),
    ):
        import app.mlflow_stub as m

        original = m._TRACKING_URI
        m._TRACKING_URI = "http://localhost:5000"
        try:
            result = log_training_run({"lr": 0.01}, {"r2": 0.85})
        finally:
            m._TRACKING_URI = original
    assert result is None


# --- AWS stub tests ---


def test_upload_no_bucket_returns_empty():
    with patch.dict("os.environ", {"S3_BUCKET": ""}):
        import app.aws_stub as s

        original = s._BUCKET
        s._BUCKET = ""
        result = upload_model_artefacts(["model.joblib"])
        s._BUCKET = original
    assert result == []


def test_upload_missing_file_skipped(tmp_path):
    import app.aws_stub as s

    original_bucket = s._BUCKET
    s._BUCKET = "test-bucket"
    mock_client = MagicMock()
    with patch("app.aws_stub._s3_client", return_value=mock_client):
        result = upload_model_artefacts([str(tmp_path / "nonexistent.joblib")])
    s._BUCKET = original_bucket
    assert result == []
    mock_client.upload_file.assert_not_called()


def test_upload_existing_file(tmp_path):
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


def test_download_no_bucket_returns_empty():
    import app.aws_stub as s

    original = s._BUCKET
    s._BUCKET = ""
    result = download_model_artefacts("/tmp")
    s._BUCKET = original
    assert result == []
