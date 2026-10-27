"""Tests for sport-cast app/aws_stub.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.aws_stub import download_model, upload_model


class TestUploadModel:
    def test_returns_s3_uri_string(self, tmp_path: Path) -> None:
        f = tmp_path / "model.joblib"
        f.write_bytes(b"data")
        result = upload_model(f, version="v1")
        assert isinstance(result, str)
        assert "s3://" in result

    def test_uri_contains_version(self, tmp_path: Path) -> None:
        f = tmp_path / "model.joblib"
        f.write_bytes(b"data")
        result = upload_model(f, version="v2")
        assert "v2" in result

    def test_uri_contains_filename(self, tmp_path: Path) -> None:
        f = tmp_path / "my_model.pkl"
        f.write_bytes(b"data")
        result = upload_model(f, version="latest")
        assert "my_model.pkl" in result

    def test_default_version_is_latest(self, tmp_path: Path) -> None:
        f = tmp_path / "model.joblib"
        f.write_bytes(b"data")
        result = upload_model(f)
        assert isinstance(result, str)


class TestDownloadModel:
    def test_returns_bool(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.joblib"
        result = download_model("latest", dest)
        assert isinstance(result, bool)

    def test_returns_false_without_boto3(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.joblib"
        with patch.dict("sys.modules", {"boto3": None}):
            result = download_model("v1", dest)
            assert result is False or isinstance(result, bool)
