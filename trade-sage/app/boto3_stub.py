"""AWS S3 model artifact stub for cloud deployment.

Provides upload/download helpers that fall back gracefully when boto3 is
unavailable or AWS credentials are not configured.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

AWS_BUCKET = os.getenv("AWS_MODEL_BUCKET", "")
AWS_PREFIX = os.getenv("AWS_MODEL_PREFIX", "trade-sage/models")

try:
    import boto3  # type: ignore[import]
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False


def upload_model(local_path: str, s3_key: str | None = None) -> str | None:
    """Upload a model artifact to S3.

    Args:
        local_path: Path to the local file to upload.
        s3_key: Override the S3 key; defaults to AWS_MODEL_PREFIX/filename.

    Returns:
        S3 URI of the uploaded artifact, or None on failure/missing config.
    """
    if not _BOTO3_AVAILABLE or not AWS_BUCKET:
        logger.debug("S3 upload skipped: boto3=%s, bucket=%s", _BOTO3_AVAILABLE, bool(AWS_BUCKET))
        return None
    key = s3_key or f"{AWS_PREFIX}/{Path(local_path).name}"
    try:
        boto3.client("s3").upload_file(local_path, AWS_BUCKET, key)
        uri = f"s3://{AWS_BUCKET}/{key}"
        logger.info("Model uploaded to %s", uri)
        return uri
    except Exception as exc:
        logger.warning("S3 upload failed: %s", exc)
        return None


def download_model(s3_key: str, local_path: str) -> bool:
    """Download a model artifact from S3.

    Args:
        s3_key: S3 object key to download.
        local_path: Destination path on the local filesystem.

    Returns:
        True on success, False otherwise.
    """
    if not _BOTO3_AVAILABLE or not AWS_BUCKET:
        return False
    try:
        boto3.client("s3").download_file(AWS_BUCKET, s3_key, local_path)
        logger.info("Model downloaded from s3://%s/%s → %s", AWS_BUCKET, s3_key, local_path)
        return True
    except Exception as exc:
        logger.warning("S3 download failed: %s", exc)
        return False
