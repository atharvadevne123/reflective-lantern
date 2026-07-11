"""AWS S3 stub — serialises model artefacts to local disk when boto3 absent."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_model(local_path: str, bucket: str, key: str) -> bool:
    """Upload a model artefact to S3, falling back to a local mirror when boto3 is absent.

    Args:
        local_path: Filesystem path to the model file to upload.
        bucket: S3 bucket name.
        key: S3 object key (path within the bucket).

    Returns:
        True on success, False if the source file does not exist.
    """
    src = Path(local_path)
    if not src.exists():
        logger.error("Model file not found: %s", local_path)
        return False
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file(str(src), bucket, key)
        logger.info("Uploaded %s → s3://%s/%s", local_path, bucket, key)
    except ImportError:
        mirror = Path("s3_mirror") / bucket / key
        mirror.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, mirror)
        logger.info("boto3 absent — mirrored %s → %s", local_path, mirror)
    return True


def download_model(bucket: str, key: str, local_path: str) -> bool:
    """Download a model artefact from S3, falling back to a local mirror when boto3 is absent.

    Args:
        bucket: S3 bucket name.
        key: S3 object key.
        local_path: Local filesystem destination path.

    Returns:
        True on success, False if neither S3 nor the mirror has the file.
    """
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.download_file(bucket, key, local_path)
        logger.info("Downloaded s3://%s/%s → %s", bucket, key, local_path)
        return True
    except ImportError:
        mirror = Path("s3_mirror") / bucket / key
        if mirror.exists():
            shutil.copy(mirror, local_path)
            logger.info("boto3 absent — copied %s → %s", mirror, local_path)
            return True
    return False
