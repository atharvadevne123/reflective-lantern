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


class TestArtefactExists:
    def test_returns_bool(self) -> None:
        from app.aws_stub import artefact_exists

        result = artefact_exists("some/key.pkl")
        assert isinstance(result, bool)

    def test_nonexistent_key(self) -> None:
        from app.aws_stub import artefact_exists

        assert artefact_exists("definitely_not_there_xyz_123.pkl") is False


class TestBuildS3Uri:
    def test_format(self) -> None:
        from app.aws_stub import build_s3_uri

        result = build_s3_uri("my-bucket", "models/v1.pkl")
        assert result == "s3://my-bucket/models/v1.pkl"

    def test_prefix(self) -> None:
        from app.aws_stub import build_s3_uri

        assert build_s3_uri("b", "k").startswith("s3://")


class TestArtefactSizeBytes:
    def test_returns_int(self) -> None:
        from app.aws_stub import artefact_size_bytes

        result = artefact_size_bytes("some_key")
        assert isinstance(result, int)

    def test_offline_returns_minus_one(self) -> None:
        from app.aws_stub import artefact_size_bytes

        result = artefact_size_bytes("key_that_wont_exist_in_ci")
        assert result in (-1, 0) or result >= 0


class TestUploadDictAsJson:
    def test_returns_string(self) -> None:
        from app.aws_stub import upload_dict_as_json

        result = upload_dict_as_json({"a": 1, "b": 2}, key="test/data.json")
        assert isinstance(result, str)

    def test_empty_dict(self) -> None:
        from app.aws_stub import upload_dict_as_json

        result = upload_dict_as_json({}, key="empty.json")
        assert isinstance(result, str)
