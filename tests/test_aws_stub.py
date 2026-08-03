"""Tests for AWS/boto3 stub."""

from __future__ import annotations

import pytest

from app.aws_stub import download_model, list_model_versions, upload_model


class TestAwsStub:
    def test_upload_returns_stub_uri(self, tmp_path):
        model_file = tmp_path / "model.joblib"
        model_file.write_text("dummy")
        result = upload_model(str(model_file))
        assert result.endswith("(stub)")
        assert "s3://" in result

    def test_download_returns_path_when_stub(self, tmp_path):
        local = str(tmp_path / "downloaded.joblib")
        result = download_model("models/v1.joblib", local)
        assert result == local

    def test_list_versions_stub(self):
        versions = list_model_versions()
        assert isinstance(versions, list)
        assert len(versions) >= 1
        assert "stub" in versions[0]

    def test_upload_missing_file_returns_stub_uri(self, tmp_path):
        result = upload_model(str(tmp_path / "nonexistent.joblib"))
        assert result.endswith("(stub)")
        assert "s3://" in result

    def test_upload_custom_bucket_appears_in_uri(self, tmp_path):
        model_file = tmp_path / "model.joblib"
        model_file.write_text("data")
        result = upload_model(str(model_file), bucket="custom-bucket")
        assert "custom-bucket" in result

    def test_download_nonexistent_key_returns_local_path(self, tmp_path):
        local = str(tmp_path / "out.joblib")
        result = download_model("models/doesnt-exist.joblib", local)
        assert result == local

    def test_list_versions_returns_list(self):
        result = list_model_versions()
        assert isinstance(result, list)

    def test_upload_returns_string(self, tmp_path):
        model_file = tmp_path / "model2.joblib"
        model_file.write_text("payload")
        result = upload_model(str(model_file))
        assert isinstance(result, str)

    @pytest.mark.parametrize("key", ["models/v1.joblib", "models/v2.joblib", "archive/v0.joblib"])
    def test_download_any_key_returns_local_path(self, tmp_path, key):
        local = str(tmp_path / "out.joblib")
        result = download_model(key, local)
        assert result == local
