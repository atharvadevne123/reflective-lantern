"""S3 model-artifact storage, with a local filesystem fallback.

Deployments that set ``S3_BUCKET`` push trained artifacts to object storage so
that a replacement replica loads the same model the trainer produced, rather
than retraining its own. Without boto3 or a bucket, every call degrades to a
no-op and the caller keeps using the local file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "cyber-guard/models")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    _BOTO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without boto3
    _BOTO_AVAILABLE = False
    BotoCoreError = ClientError = Exception


def is_s3_enabled() -> bool:
    """Return True when boto3 is importable and a bucket is configured."""
    return _BOTO_AVAILABLE and bool(S3_BUCKET)


def _client():
    """Build an S3 client for the configured region."""
    return boto3.client("s3", region_name=AWS_REGION)


def s3_key_for(local_path: str) -> str:
    """Return the object key an artifact would occupy in the bucket.

    Args:
        local_path: Path to the local artifact.

    Returns:
        The ``prefix/basename`` key.
    """
    return f"{S3_PREFIX}/{Path(local_path).name}"


def upload_artifact(local_path: str) -> str | None:
    """Upload a model artifact to S3.

    Args:
        local_path: Path to the artifact to upload.

    Returns:
        The ``s3://`` URI on success, otherwise None (disabled, missing file,
        or an AWS error — none of which are fatal to the caller).
    """
    if not is_s3_enabled():
        logger.debug("s3 disabled, keeping artifact local: %s", local_path)
        return None
    if not Path(local_path).exists():
        logger.warning("artifact not found, nothing to upload: %s", local_path)
        return None

    key = s3_key_for(local_path)
    try:
        _client().upload_file(local_path, S3_BUCKET, key)
    except (BotoCoreError, ClientError) as exc:
        logger.error("s3 upload failed for %s: %s", key, exc)
        return None

    uri = f"s3://{S3_BUCKET}/{key}"
    logger.info("artifact uploaded %s", uri)
    return uri


def download_artifact(local_path: str) -> bool:
    """Fetch a model artifact from S3 into ``local_path``.

    Args:
        local_path: Destination path; its basename selects the object key.

    Returns:
        True when a file was written, False otherwise.
    """
    if not is_s3_enabled():
        return False

    key = s3_key_for(local_path)
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        _client().download_file(S3_BUCKET, key, local_path)
    except (BotoCoreError, ClientError) as exc:
        logger.warning("s3 download failed for %s: %s", key, exc)
        return False

    logger.info("artifact downloaded s3://%s/%s", S3_BUCKET, key)
    return True
