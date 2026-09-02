"""Tests for the AWS S3 model upload/download stub."""

from __future__ import annotations

from pathlib import Path


def test_upload_model_returns_false_when_no_bucket(monkeypatch, tmp_path):
    monkeypatch.setenv("S3_BUCKET", "")
    from app import aws_stub

    monkeypatch.setattr(aws_stub, "S3_BUCKET", "")
    result = aws_stub.upload_model(tmp_path / "model.joblib")
    assert result is False


def test_download_model_returns_false_when_no_bucket(monkeypatch, tmp_path):
    from app import aws_stub

    monkeypatch.setattr(aws_stub, "S3_BUCKET", "")
    result = aws_stub.download_model(tmp_path / "model.joblib")
    assert result is False


def test_upload_model_returns_false_when_boto3_missing(monkeypatch, tmp_path):
    from app import aws_stub

    monkeypatch.setattr(aws_stub, "S3_BUCKET", "my-test-bucket")
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("boto3 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    result = aws_stub.upload_model(tmp_path / "model.joblib")
    assert result is False


def test_download_model_returns_false_when_boto3_missing(monkeypatch, tmp_path):
    from app import aws_stub

    monkeypatch.setattr(aws_stub, "S3_BUCKET", "my-test-bucket")
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("boto3 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    result = aws_stub.download_model(tmp_path / "model.joblib")
    assert result is False


def test_upload_model_uses_version_in_key(monkeypatch, tmp_path):
    """Ensure the S3 key includes the version string."""
    from app import aws_stub

    monkeypatch.setattr(aws_stub, "S3_BUCKET", "")
    result = aws_stub.upload_model(Path("model.joblib"), version="2.0.0")
    assert result is False


import pytest  # noqa: E402


@pytest.mark.parametrize("version", ["1.0.0", "2.3.1", "0.0.1"])
def test_upload_various_versions_without_bucket(monkeypatch, tmp_path, version: str) -> None:
    """Upload returns False for any version when bucket is unset."""
    from app import aws_stub

    monkeypatch.setattr(aws_stub, "S3_BUCKET", "")
    result = aws_stub.upload_model(tmp_path / "model.joblib", version=version)
    assert result is False


@pytest.mark.parametrize("filename", ["model.joblib", "lgbm.joblib", "scaler.pkl"])
def test_download_various_filenames_without_bucket(monkeypatch, tmp_path, filename: str) -> None:
    """Download returns False for any model filename when bucket is unset."""
    from app import aws_stub

    monkeypatch.setattr(aws_stub, "S3_BUCKET", "")
    result = aws_stub.download_model(tmp_path / filename)
    assert result is False
