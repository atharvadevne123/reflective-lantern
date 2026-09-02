"""Tests for MLflow tracking and S3 storage graceful degradation.

Both integrations are optional. The contract these tests pin is that a
missing dependency, an unset configuration, or a backend error never
propagates out to the caller.
"""

from __future__ import annotations

import pytest

from app import storage, tracking

# --- tracking ---

def test_tracking_disabled_without_uri(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    assert tracking.is_tracking_enabled() is False


def test_track_run_is_noop_when_disabled(monkeypatch):
    """The context manager must still yield when tracking is off."""
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    entered = False
    with tracking.track_run("test"):
        entered = True
    assert entered


def test_log_metrics_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    tracking.log_metrics({"accuracy": 0.9})  # must not raise


def test_log_params_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    tracking.log_params({"depth": 4})  # must not raise


# --- storage ---

def test_s3_disabled_without_bucket(monkeypatch):
    monkeypatch.setattr(storage, "S3_BUCKET", "")
    assert storage.is_s3_enabled() is False


def test_upload_returns_none_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "S3_BUCKET", "")
    f = tmp_path / "model.joblib"
    f.write_text("x")
    assert storage.upload_artifact(str(f)) is None


def test_download_returns_false_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "S3_BUCKET", "")
    assert storage.download_artifact(str(tmp_path / "model.joblib")) is False


def test_s3_key_uses_prefix_and_basename(monkeypatch):
    monkeypatch.setattr(storage, "S3_PREFIX", "cyber-guard/models")
    assert storage.s3_key_for("/tmp/deep/model.joblib") == "cyber-guard/models/model.joblib"


def test_upload_missing_file_returns_none(monkeypatch):
    """A missing artifact is logged and skipped, not raised."""
    monkeypatch.setattr(storage, "S3_BUCKET", "some-bucket")
    monkeypatch.setattr(storage, "_BOTO_AVAILABLE", True)
    assert storage.upload_artifact("/nonexistent/model.joblib") is None


def test_upload_swallows_aws_error(monkeypatch, tmp_path):
    """An AWS failure must not fail the training run that triggered it."""
    monkeypatch.setattr(storage, "S3_BUCKET", "some-bucket")
    monkeypatch.setattr(storage, "_BOTO_AVAILABLE", True)

    class _Boom:
        def upload_file(self, *a, **k):
            raise storage.ClientError

    monkeypatch.setattr(storage, "_client", lambda: _Boom())
    f = tmp_path / "model.joblib"
    f.write_text("x")
    assert storage.upload_artifact(str(f)) is None


@pytest.mark.parametrize("metrics", [
    {"accuracy": 0.9, "classes": ["a", "b"]},
    {"auc": float("nan")},
    {},
])
def test_log_metrics_tolerates_non_numeric(monkeypatch, metrics):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    tracking.log_metrics(metrics)  # must not raise
