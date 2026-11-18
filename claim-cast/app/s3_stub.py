"""AWS S3 stub for model artifact storage — replace with boto3 in production."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "claim-cast-artifacts")
S3_PREFIX = os.getenv("S3_PREFIX", "models")


def upload_artifact(local_path: str | Path, key: str | None = None) -> str:
    """Upload a local artifact to S3 (stub logs the action).

    Args:
        local_path: Local file path to upload.
        key: S3 object key. Defaults to the filename.

    Returns:
        s3:// URI string where the artifact would live.
    """
    path = Path(local_path)
    if key is None:
        key = f"{S3_PREFIX}/{path.name}"
    s3_uri = f"s3://{S3_BUCKET}/{key}"
    logger.info("s3_upload_stub local=%s uri=%s size_bytes=%d", path, s3_uri, path.stat().st_size if path.exists() else 0)
    return s3_uri


def download_artifact(s3_uri: str, local_path: str | Path) -> Path:
    """Download an artifact from S3 (stub — just validates the URI format).

    Args:
        s3_uri: s3:// URI of the artifact.
        local_path: Local destination path.

    Returns:
        Path object pointing to local_path.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    logger.info("s3_download_stub uri=%s local=%s", s3_uri, local_path)
    return Path(local_path)


def list_artifacts(prefix: str | None = None) -> list[str]:
    """List artifact keys in the S3 bucket (stub returns empty list).

    Args:
        prefix: Optional key prefix filter.

    Returns:
        List of S3 object key strings.
    """
    effective_prefix = prefix or S3_PREFIX
    logger.info("s3_list_stub bucket=%s prefix=%s", S3_BUCKET, effective_prefix)
    return []
