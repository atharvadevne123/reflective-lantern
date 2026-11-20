"""Extended tests for boto3_stub S3 utilities."""

from __future__ import annotations

from unittest.mock import patch

from app.boto3_stub import is_s3_configured, list_model_artifacts


class TestIsS3Configured:
    def test_returns_false_when_boto3_unavailable(self) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", False):
            assert is_s3_configured() is False

    def test_returns_false_when_no_bucket(self) -> None:
        with patch("app.boto3_stub.AWS_BUCKET", ""):
            assert is_s3_configured() is False

    def test_returns_true_when_both_available(self) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", True), \
             patch("app.boto3_stub.AWS_BUCKET", "my-bucket"):
            assert is_s3_configured() is True

    def test_returns_bool(self) -> None:
        result = is_s3_configured()
        assert isinstance(result, bool)


class TestListModelArtifacts:
    def test_empty_list_when_not_configured(self) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", False):
            assert list_model_artifacts() == []

    def test_empty_list_when_no_bucket(self) -> None:
        with patch("app.boto3_stub.AWS_BUCKET", ""):
            assert list_model_artifacts() == []

    def test_returns_list_type(self) -> None:
        result = list_model_artifacts()
        assert isinstance(result, list)

    def test_custom_prefix_accepted(self) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", False):
            result = list_model_artifacts(prefix="custom/prefix/")
            assert result == []

    def test_s3_error_returns_empty(self) -> None:
        with patch("app.boto3_stub._BOTO3_AVAILABLE", True), \
             patch("app.boto3_stub.AWS_BUCKET", "my-bucket"), \
             patch("app.boto3_stub.boto3") as mock_boto3:
            mock_boto3.client.side_effect = RuntimeError("no credentials")
            result = list_model_artifacts()
            assert result == []
