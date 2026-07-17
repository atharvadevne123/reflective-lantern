"""AWS S3 stub — returns stub responses when boto3/S3 are unavailable."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"
DEFAULT_PREFIX = "realty-edge/models"
ARTEFACT_FILENAMES = ("model.joblib", "metrics.json")

_REGION = os.getenv("AWS_REGION", DEFAULT_REGION)
_BUCKET = os.getenv("S3_BUCKET", "watt-guard-models")
_PREFIX = os.getenv("S3_PREFIX", DEFAULT_PREFIX)


def _s3_client() -> Any | None:
    """Return a boto3 S3 client, or None when boto3 is absent or no bucket is configured."""
    if not _BUCKET:
        return None
    try:
        import boto3

        return boto3.client("s3", region_name=_REGION)
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("Failed to create S3 client: %s", exc)
        return None


def upload_model(local_path: str, bucket: str = "", key: str = "") -> str:
    """Upload a model artefact to S3, returning a stub URI when boto3 is absent.

    Args:
        local_path: Filesystem path to the model file to upload.
        bucket: S3 bucket name (defaults to env S3_BUCKET).
        key: S3 object key (defaults to PREFIX/filename).

    Returns:
        S3 URI string on success, or a stub URI when boto3 is unavailable.
    """
    bucket = bucket or _BUCKET or "stub-bucket"
    src = Path(local_path)
    key = key or f"{_PREFIX}/{src.name}"
    stub_uri = f"s3://{bucket}/{key} (stub)"
    if not src.exists():
        logger.error("Model file not found: %s", local_path)
        return stub_uri
    client = _s3_client()
    if client is None:
        mirror = Path("s3_mirror") / bucket / key
        mirror.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, mirror)
        logger.info("boto3 absent — mirrored %s → %s", local_path, mirror)
        return stub_uri
    try:
        client.upload_file(str(src), bucket, key)
        uri = f"s3://{bucket}/{key}"
        logger.info("Uploaded %s → %s", local_path, uri)
        return uri
    except Exception as exc:
        logger.error("S3 upload failed: %s", exc)
        return stub_uri


def download_model(key: str, local_path: str, bucket: str = "") -> str:
    """Download a model artefact from S3, falling back to local mirror when boto3 absent.

    Args:
        key: S3 object key to download.
        local_path: Local filesystem destination path.
        bucket: S3 bucket name (defaults to env S3_BUCKET).

    Returns:
        Local path on success or when stub; empty string on hard failure.
    """
    bucket = bucket or _BUCKET or "stub-bucket"
    client = _s3_client()
    if client is None:
        mirror = Path("s3_mirror") / bucket / key
        if mirror.exists():
            shutil.copy(mirror, local_path)
            logger.info("Restored from mirror %s", mirror)
        else:
            logger.debug("boto3 absent and no mirror; returning stub path %s", local_path)
        return local_path
    try:
        client.download_file(bucket, key, local_path)
        logger.info("Downloaded s3://%s/%s → %s", bucket, key, local_path)
        return local_path
    except Exception as exc:
        logger.error("S3 download failed for %s: %s", key, exc)
        return local_path


def list_model_versions(bucket: str = "", prefix: str = "") -> list[str]:
    """List model version keys available in S3.

    Args:
        bucket: S3 bucket name (defaults to env S3_BUCKET).
        prefix: Key prefix to filter (defaults to env S3_PREFIX).

    Returns:
        List of S3 key strings; a single stub entry when boto3 is absent.
    """
    bucket = bucket or _BUCKET or "stub-bucket"
    prefix = prefix or _PREFIX
    client = _s3_client()
    if client is None:
        return [f"{prefix}/model.joblib (stub)"]
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        return keys if keys else [f"{prefix}/model.joblib (stub)"]
    except Exception as exc:
        logger.error("S3 list failed: %s", exc)
        return [f"{prefix}/model.joblib (stub)"]


def upload_model_artefacts(local_paths: list[str]) -> list[str]:
    """Upload multiple model artefacts to S3.

    Args:
        local_paths: Local file paths to upload.

    Returns:
        List of S3 URIs that were successfully uploaded; empty when no bucket or files missing.
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
    for filename in ARTEFACT_FILENAMES:
        key = f"{_PREFIX}/{filename}"
        local_path = os.path.join(local_dir, filename)
        try:
            client.download_file(_BUCKET, key, local_path)
            downloaded.append(local_path)
            logger.info("Downloaded %s to %s", key, local_path)
        except Exception as exc:
            logger.warning("S3 download failed for %s: %s", key, exc)
    return downloaded


def delete_artefact(key: str, bucket: str = "") -> bool:
    """Delete a model artefact from S3.

    Args:
        key: S3 object key to delete.
        bucket: S3 bucket name (defaults to env S3_BUCKET).

    Returns:
        True on success, False when boto3 absent or deletion fails.
    """
    bucket = bucket or _BUCKET
    if not bucket:
        return False
    client = _s3_client()
    if client is None:
        return False
    try:
        client.delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted s3://%s/%s", bucket, key)
        return True
    except Exception as exc:
        logger.error("S3 delete failed for %s: %s", key, exc)
        return False


__all__ = [
    "ARTEFACT_FILENAMES",
    "DEFAULT_PREFIX",
    "DEFAULT_REGION",
    "delete_artefact",
    "download_model",
    "download_model_artefacts",
    "list_model_versions",
    "upload_model",
    "upload_model_artefacts",
]
