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
