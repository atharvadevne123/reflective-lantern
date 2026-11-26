"""AWS S3 stub for model artifact upload and download."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import boto3

    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False


def upload_model_to_s3(local_path: str, bucket: str = "", key: str = "") -> bool:
    """Upload a model artifact to S3. Returns True on success."""
    if not _BOTO3_AVAILABLE:
        logger.warning("boto3 not installed - S3 upload skipped")
        return False
    bucket = bucket or os.getenv("S3_BUCKET", "")
    if not bucket:
        logger.warning("S3_BUCKET not set - skipping upload")
        return False
    s3 = boto3.client("s3")
    key = key or Path(local_path).name
    s3.upload_file(local_path, bucket, key)
    logger.info("Uploaded %s -> s3://%s/%s", local_path, bucket, key)
    return True


def download_model_from_s3(bucket: str, key: str, local_path: str) -> bool:
    """Download a model artifact from S3. Returns True on success."""
    if not _BOTO3_AVAILABLE:
        logger.warning("boto3 not installed - S3 download skipped")
        return False
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    logger.info("Downloaded s3://%s/%s -> %s", bucket, key, local_path)
    return True
