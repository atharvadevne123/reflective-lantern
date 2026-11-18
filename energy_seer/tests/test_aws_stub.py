"""Tests for Energy-Seer AWS S3 stub."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.aws_stub import (
    delete_model,
    download_model,
    list_model_versions,
    model_exists,
    upload_model,
)


def test_upload_model_returns_true():
    result = upload_model("/tmp/model.joblib", "my-bucket", "models/m.joblib")
    assert result is True


def test_download_model_returns_true():
    result = download_model("my-bucket", "models/m.joblib", "/tmp/model.joblib")
    assert result is True


def test_list_model_versions_returns_list():
    result = list_model_versions("my-bucket")
    assert isinstance(result, list)


def test_list_model_versions_stub_returns_empty():
    result = list_model_versions("my-bucket", prefix="models/")
    assert result == []


def test_delete_model_returns_true():
    result = delete_model("my-bucket", "models/m.joblib")
    assert result is True


def test_model_exists_stub_returns_bool():
    result = model_exists("my-bucket", "models/m.joblib")
    assert isinstance(result, bool)


def test_upload_accepts_path_object():
    result = upload_model(Path("/tmp/model.joblib"), "bucket", "key")
    assert result is True


def test_download_accepts_path_object():
    result = download_model("bucket", "key", Path("/tmp/model.joblib"))
    assert result is True


@pytest.mark.parametrize(
    "bucket,key",
    [
        ("prod-bucket", "models/v1.joblib"),
        ("dev-bucket", "models/latest.joblib"),
    ],
)
def test_upload_various_buckets(bucket, key):
    result = upload_model("/tmp/m.joblib", bucket, key)
    assert result is True
