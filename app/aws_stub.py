"""AWS/boto3 integration stub for Realty-Edge model artefact storage.

Uploads and downloads model artefacts (model.joblib, metrics.json) to an
S3 bucket when AWS credentials are configured. Falls back to local disk
when AWS is unavailable.

Environment variables
---------------------
AWS_REGION          – AWS region (default: us-east-1)
S3_BUCKET           – Target bucket name
S3_PREFIX           – Key prefix for artefacts (default: realty-edge/models)
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REGION = os.getenv("AWS_REGION", "us-east-1")
_BUCKET = os.getenv("S3_BUCKET", "")
_PREFIX = os.getenv("S3_PREFIX", "realty-edge/models")


def _s3_client() -> Any | None:
    """Return a boto3 S3 client or None if boto3 is unavailable."""
    if not _BUCKET:
        return None
    try:
        import boto3

        return boto3.client("s3", region_name=_REGION)
    except ImportError:
        logger.debug("boto3 not installed — S3 artefact storage disabled.")
    except Exception as exc:
        logger.warning("Failed to initialise boto3 client: %s", exc)
    return None


def upload_model_artefacts(local_paths: list[str]) -> list[str]:
    """Upload model artefact files to S3.

    Args:
        local_paths: Local file paths to upload.

    Returns:
        List of S3 URIs that were successfully uploaded.
    """
    client = _s3_client()
    if client is None:
        logger.debug("S3 upload skipped (no bucket configured or boto3 missing).")
        return []
    uploaded: list[str] = []
    for path in local_paths:
        if not Path(path).exists():
            logger.warning("Artefact not found, skipping upload: %s", path)
            continue
        key = f"{_PREFIX}/{Path(path).name}"
        try:
            client.upload_file(path, _BUCKET, key)
            uri = f"s3://{_BUCKET}/{key}"
            uploaded.append(uri)
            logger.info("Uploaded artefact to %s", uri)
        except Exception as exc:
            logger.error("S3 upload failed for %s: %s", path, exc)
    return uploaded


def download_model_artefacts(local_dir: str = ".") -> list[str]:
    """Download model artefacts from S3 to a local directory.

    Args:
        local_dir: Directory to write downloaded files into.

    Returns:
        List of local file paths that were successfully downloaded.
    """
    client = _s3_client()
    if client is None:
        logger.debug("S3 download skipped.")
        return []
    downloaded: list[str] = []
    for filename in ("model.joblib", "metrics.json"):
        key = f"{_PREFIX}/{filename}"
        local_path = os.path.join(local_dir, filename)
        try:
            client.download_file(_BUCKET, key, local_path)
            downloaded.append(local_path)
            logger.info("Downloaded %s to %s", key, local_path)
        except Exception as exc:
            logger.warning("S3 download failed for %s: %s", key, exc)
    return downloaded
