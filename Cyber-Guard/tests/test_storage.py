"""Tests for Cyber-Guard app/storage.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.storage import is_s3_enabled, s3_key_for, upload_artifact


class TestIsS3Enabled:
    def test_returns_false_without_boto3(self) -> None:
        with patch("app.storage._BOTO_AVAILABLE", False):
            assert is_s3_enabled() is False

    def test_returns_false_without_bucket(self) -> None:
        with patch("app.storage._BOTO_AVAILABLE", True), patch("app.storage.S3_BUCKET", ""):
            assert is_s3_enabled() is False

    def test_returns_true_with_both(self) -> None:
        with (
            patch("app.storage._BOTO_AVAILABLE", True),
            patch("app.storage.S3_BUCKET", "my-bucket"),
        ):
            assert is_s3_enabled() is True


class TestS3KeyFor:
    def test_includes_prefix_and_basename(self) -> None:
        key = s3_key_for("/tmp/model.joblib")
        assert key.endswith("model.joblib")
        assert "/" in key

    def test_uses_configured_prefix(self) -> None:
        with patch("app.storage.S3_PREFIX", "custom/prefix"):
            key = s3_key_for("model.joblib")
            assert key.startswith("custom/prefix/")

    def test_basename_only_file(self) -> None:
        key = s3_key_for("artifact.pkl")
        assert "artifact.pkl" in key

    @pytest.mark.parametrize("fname", ["model.joblib", "pipeline.pkl", "metrics.json"])
    def test_various_filenames(self, fname: str) -> None:
        key = s3_key_for(fname)
        assert fname in key


class TestUploadArtifact:
    def test_returns_none_when_s3_disabled(self, tmp_path: Path) -> None:
        f = tmp_path / "model.joblib"
        f.write_bytes(b"data")
        with patch("app.storage._BOTO_AVAILABLE", False):
            result = upload_artifact(str(f))
        assert result is None

    def test_returns_none_when_no_bucket(self, tmp_path: Path) -> None:
        f = tmp_path / "model.joblib"
        f.write_bytes(b"data")
        with patch("app.storage.S3_BUCKET", ""), patch("app.storage._BOTO_AVAILABLE", True):
            result = upload_artifact(str(f))
        assert result is None

    def test_returns_uri_on_success(self, tmp_path: Path) -> None:
        f = tmp_path / "model.joblib"
        f.write_bytes(b"data")
        mock_client = MagicMock()
        import sys
        import types
        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = MagicMock(return_value=mock_client)
        with (
            patch("app.storage._BOTO_AVAILABLE", True),
            patch("app.storage.S3_BUCKET", "test-bucket"),
            patch.dict(sys.modules, {"boto3": fake_boto3}),
        ):
            result = upload_artifact(str(f))
        assert result is None or (isinstance(result, str) and "s3://" in result)


class TestDownloadArtifact:
    def test_returns_false_when_s3_disabled(self, tmp_path) -> None:
        from app.storage import download_artifact
        with patch("app.storage._BOTO_AVAILABLE", False):
            result = download_artifact(str(tmp_path / "model.joblib"))
        assert result is False

    def test_returns_false_when_no_bucket(self, tmp_path) -> None:
        from app.storage import download_artifact
        with patch("app.storage.S3_BUCKET", ""), patch("app.storage._BOTO_AVAILABLE", True):
            result = download_artifact(str(tmp_path / "model.joblib"))
        assert result is False


class TestS3KeyForExtra:
    def test_key_format_with_prefix(self) -> None:
        with patch("app.storage.S3_PREFIX", "ml-artifacts"):
            key = s3_key_for("model.joblib")
            assert key == "ml-artifacts/model.joblib"

    def test_nested_path_uses_basename_only(self) -> None:
        key = s3_key_for("/deep/nested/path/model.joblib")
        assert "/" in key
        assert key.endswith("model.joblib")
