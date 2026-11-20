"""AWS S3 / boto3 stub for model artifact upload — swappable for real boto3."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_model(local_path: str | Path, bucket: str, key: str) -> bool:
    """Upload a model artifact to S3 (stub logs instead of calling S3)."""
    logger.info("s3_upload bucket=%s key=%s path=%s", bucket, key, local_path)
    return True


def download_model(bucket: str, key: str, local_path: str | Path) -> bool:
    """Download a model artifact from S3 (stub logs instead of calling S3)."""
    logger.info("s3_download bucket=%s key=%s dest=%s", bucket, key, local_path)
    return True


def list_model_versions(bucket: str, prefix: str = "models/") -> list[str]:
    """List model versions stored in S3."""
    logger.info("s3_list bucket=%s prefix=%s", bucket, prefix)
    return []


def delete_model(bucket: str, key: str) -> bool:
    """Delete a model artifact from S3 (stub logs instead of calling S3).

    Args:
        bucket: S3 bucket name.
        key: Object key to delete.

    Returns:
        True (stub always succeeds).
    """
    logger.info("s3_delete bucket=%s key=%s", bucket, key)
    return True


def model_exists(bucket: str, key: str) -> bool:
    """Check whether a model artifact exists in S3 (stub always returns False).

    Args:
        bucket: S3 bucket name.
        key: Object key to check.

    Returns:
        False (stub — no real S3 lookup performed).
    """
    logger.info("s3_exists bucket=%s key=%s", bucket, key)
    return False


__all__ = [
    "delete_model",
    "download_model",
    "list_model_versions",
    "model_exists",
    "upload_model",
]
