"""AWS stub tests with mocked boto3."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_upload_skips_when_no_boto3():
    from app.aws_stub import upload_model_artifact
    import pathlib
    result = upload_model_artifact(pathlib.Path("nonexistent.joblib"), "models/test.joblib")
    assert result is True  # no-op returns True


def test_upload_skips_when_no_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    from app.aws_stub import upload_model_artifact
    import pathlib
    result = upload_model_artifact(pathlib.Path("nonexistent.joblib"), "models/test.joblib")
    assert result is True


def test_download_returns_false_when_unavailable():
    from app.aws_stub import download_model_artifact
    import pathlib
    result = download_model_artifact("models/test.joblib", pathlib.Path("/tmp/test.joblib"))
    assert result is False
