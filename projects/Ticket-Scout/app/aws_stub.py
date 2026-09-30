"""AWS/boto3 stub for S3 model artifact storage and SQS ticket ingestion."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_BOTO3_AVAILABLE = False
try:
    import boto3  # type: ignore[import]
    _BOTO3_AVAILABLE = True
except ImportError:
    pass

S3_BUCKET = os.getenv("S3_BUCKET", "ticket-scout-models")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def upload_model_artifact(local_path: Path, s3_key: str) -> bool:
    """Upload a model file to S3 if boto3 is configured.

    Args:
        local_path: Local file to upload.
        s3_key: Destination key in S3_BUCKET.

    Returns:
        True if upload succeeded or boto3 is unavailable (no-op).
    """
    if not _BOTO3_AVAILABLE:
        logger.info("boto3 not available — skipping S3 upload of %s", local_path)
        return True

    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    if not aws_key:
        logger.info("AWS credentials not set — skipping S3 upload")
        return True

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)
        logger.info("Uploaded %s to s3://%s/%s", local_path, S3_BUCKET, s3_key)
        return True
    except Exception as exc:
        logger.error("S3 upload failed: %s", exc)
        return False


def download_model_artifact(s3_key: str, local_path: Path) -> bool:
    """Download a model artifact from S3.

    Args:
        s3_key: Source key in S3_BUCKET.
        local_path: Destination local path.

    Returns:
        True if download succeeded, False otherwise.
    """
    if not _BOTO3_AVAILABLE:
        return False

    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    if not aws_key:
        return False

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.download_file(S3_BUCKET, s3_key, str(local_path))
        logger.info("Downloaded s3://%s/%s to %s", S3_BUCKET, s3_key, local_path)
        return True
    except Exception as exc:
        logger.error("S3 download failed: %s", exc)
        return False
