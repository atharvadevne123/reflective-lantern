"""Tests for trade-sage app/boto3_stub.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.boto3_stub import download_model, upload_model


class TestUploadModel:
    def test_returns_none_without_bucket(self) -> None:
        with patch("app.boto3_stub.AWS_BUCKET", ""):
            result = upload_model("/tmp/model.joblib")
        assert result is None

    def test_returns_none_without_boto3(self) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", False):
            result = upload_model("/tmp/model.joblib")
        assert result is None

    def test_accepts_s3_key_override(self) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", False):
            result = upload_model("/tmp/model.joblib", s3_key="custom/key/model.joblib")
        assert result is None

    @pytest.mark.parametrize("path", ["/tmp/a.joblib", "/tmp/b.pkl", "/tmp/c.json"])
    def test_various_paths(self, path: str) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", False):
            result = upload_model(path)
        assert result is None or isinstance(result, str)


class TestDownloadModel:
    def test_returns_false_without_bucket(self) -> None:
        with patch("app.boto3_stub.AWS_BUCKET", ""):
            result = download_model("model/key", "/tmp/out.joblib")
        assert result is False

    def test_returns_false_without_boto3(self) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", False):
            result = download_model("model/key", "/tmp/out.joblib")
        assert result is False

    def test_returns_bool(self) -> None:
        result = download_model("model/key", "/tmp/out.joblib")
        assert isinstance(result, bool)
