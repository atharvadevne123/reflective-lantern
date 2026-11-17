"""Tests for Sports-Oracle AWS stub upload/download with boto3 mocked."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.aws_stub import download_model, upload_model


def test_upload_returns_bool():
    with patch("builtins.__import__", side_effect=ImportError):
        # ImportError path -> returns False
        pass
    # Without boto3 installed (or simulated ImportError), stub returns False
    result = upload_model(Path("/tmp/model.joblib"), "my-bucket", "models/m.joblib")
    assert isinstance(result, bool)


def test_upload_with_mock_boto3_returns_true():
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value.upload_file = MagicMock()
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = upload_model(Path("/tmp/model.joblib"), "bucket", "key")
        assert result is True


def test_upload_with_boto3_exception_returns_false():
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value.upload_file.side_effect = Exception("S3 error")
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = upload_model(Path("/tmp/model.joblib"), "bucket", "key")
        assert result is False


def test_download_returns_bool():
    result = download_model("my-bucket", "models/m.joblib", Path("/tmp/model.joblib"))
    assert isinstance(result, bool)


def test_download_with_mock_boto3_returns_true():
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value.download_file = MagicMock()
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = download_model("bucket", "key", Path("/tmp/model.joblib"))
        assert result is True


def test_download_with_boto3_exception_returns_false():
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value.download_file.side_effect = Exception("S3 error")
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = download_model("bucket", "key", Path("/tmp/model.joblib"))
        assert result is False
