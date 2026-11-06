"""Extended tests for energy_seer app/aws_stub.py."""

from __future__ import annotations

import logging

import pytest

from app.aws_stub import delete_model, download_model, list_model_versions, model_exists, upload_model


class TestUploadModelExtended:
    def test_returns_true(self) -> None:
        assert upload_model("/path/to/model.pkl", "my-bucket", "models/v1.pkl") is True

    def test_logs_upload(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.aws_stub"):
            upload_model("/path/model.pkl", "bucket", "key")
        assert any("s3_upload" in r.message for r in caplog.records)

    def test_logs_bucket_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.aws_stub"):
            upload_model("/p", "my-special-bucket", "k")
        assert any("my-special-bucket" in r.message for r in caplog.records)

    def test_path_object_accepted(self) -> None:
        from pathlib import Path
        result = upload_model(Path("/tmp/model.pkl"), "b", "k")
        assert result is True


class TestDownloadModelExtended:
    def test_returns_true(self) -> None:
        assert download_model("bucket", "key", "/dest/model.pkl") is True

    def test_logs_download(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.aws_stub"):
            download_model("bucket", "key", "/dest")
        assert any("s3_download" in r.message for r in caplog.records)


class TestListModelVersionsExtended:
    def test_returns_empty_list(self) -> None:
        result = list_model_versions("bucket")
        assert result == []

    def test_returns_list_type(self) -> None:
        assert isinstance(list_model_versions("bucket", prefix="models/"), list)

    def test_logs_list(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.aws_stub"):
            list_model_versions("bucket", "prefix/")
        assert any("s3_list" in r.message for r in caplog.records)


class TestDeleteModelExtended:
    def test_returns_true(self) -> None:
        assert delete_model("bucket", "models/old.pkl") is True

    def test_logs_delete(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.aws_stub"):
            delete_model("bucket", "key")
        assert any("s3_delete" in r.message for r in caplog.records)

    def test_logs_key(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.aws_stub"):
            delete_model("b", "models/special-key.pkl")
        assert any("special-key.pkl" in r.message for r in caplog.records)


class TestModelExistsExtended:
    def test_returns_false(self) -> None:
        assert model_exists("bucket", "models/v1.pkl") is False

    def test_logs_exists_check(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.aws_stub"):
            model_exists("bucket", "key")
        assert any("s3_exists" in r.message for r in caplog.records)

    @pytest.mark.parametrize("key", ["v1/model.pkl", "v2/model.pkl", "latest/model.pkl"])
    def test_always_false_for_any_key(self, key: str) -> None:
        assert model_exists("bucket", key) is False
