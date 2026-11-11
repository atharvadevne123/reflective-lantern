"""AWS S3 model artefact storage stub."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import boto3  # type: ignore[import-untyped]
    _BOTO3_ENABLED = True
except ImportError:
    _BOTO3_ENABLED = False


S3_BUCKET = os.getenv("S3_MODEL_BUCKET", "alpha-scout-models")
S3_KEY_PREFIX = os.getenv("S3_KEY_PREFIX", "models/")


def upload_model(local_path: str | Path, version: str = "latest") -> bool:
    """Upload model artefact to S3; returns True on success."""
    if not _BOTO3_ENABLED:
        logger.info("boto3 not available; S3 upload skipped for %s", local_path)
        return False
    client = boto3.client("s3")
    key = f"{S3_KEY_PREFIX}{version}/model.joblib"
    try:
        client.upload_file(str(local_path), S3_BUCKET, key)
        logger.info("uploaded model to s3://%s/%s", S3_BUCKET, key)
        return True
    except Exception as exc:
        logger.error("S3 upload failed: %s", exc)
        return False


def download_model(version: str = "latest", dest: str | Path = "model.joblib") -> bool:
    """Download model artefact from S3; returns True on success."""
    if not _BOTO3_ENABLED:
        logger.info("boto3 not available; S3 download skipped")
        return False
    client = boto3.client("s3")
    key = f"{S3_KEY_PREFIX}{version}/model.joblib"
    try:
        client.download_file(S3_BUCKET, key, str(dest))
        logger.info("downloaded model from s3://%s/%s", S3_BUCKET, key)
        return True
    except Exception as exc:
        logger.error("S3 download failed: %s", exc)
        return False
