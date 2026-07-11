"""AWS S3 stub — serialises model artefacts to local disk when boto3 absent."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"
DEFAULT_PREFIX = "realty-edge/models"
ARTEFACT_FILENAMES = ("model.joblib", "metrics.json")

_REGION = os.getenv("AWS_REGION", DEFAULT_REGION)
_BUCKET = os.getenv("S3_BUCKET", "")
_PREFIX = os.getenv("S3_PREFIX", DEFAULT_PREFIX)

def upload_model(local_path: str, bucket: str, key: str) -> bool:
    """Upload model to S3, falling back to a local mirror."""
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
    """Download model from S3 or local mirror."""
    try:
        import boto3

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
