"""Tests for AWS/boto3 stub."""

from __future__ import annotations

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
