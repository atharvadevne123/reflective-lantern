"""AWS S3 stub for model artifact upload/download in Sport-Cast."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_S3_BUCKET = os.getenv("S3_BUCKET", "sport-cast-models")
_S3_PREFIX = os.getenv("S3_MODEL_PREFIX", "models/sport-cast/")


def upload_model(local_path: Path, version: str = "latest") -> str:
    """Upload a model artifact to S3 (stub: logs without uploading).

    Args:
        local_path: Path to the local model file.
        version: Version tag for the S3 key.

    Returns:
        S3 URI string (stub always returns the expected URI without uploading).
    """
    s3_key = f"{_S3_PREFIX}{version}/{local_path.name}"
    s3_uri = f"s3://{_S3_BUCKET}/{s3_key}"
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file(str(local_path), _S3_BUCKET, s3_key)
        logger.info("Uploaded %s to %s", local_path, s3_uri)
    except ImportError:
        logger.info("boto3 not installed — skipping S3 upload (stub): %s -> %s", local_path, s3_uri)
    except Exception as exc:
        logger.warning("S3 upload failed (non-fatal): %s", exc)
    return s3_uri


def download_model(version: str, local_path: Path) -> bool:
    """Download a model artifact from S3 (stub: logs without downloading).

    Args:
        version: Version tag for the S3 key.
        local_path: Destination path for the downloaded file.

    Returns:
        True if download succeeded, False otherwise.
    """
    s3_key = f"{_S3_PREFIX}{version}/model.joblib"
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.download_file(_S3_BUCKET, s3_key, str(local_path))
        logger.info("Downloaded s3://%s/%s to %s", _S3_BUCKET, s3_key, local_path)
        return True
    except ImportError:
        logger.info("boto3 not installed — skipping S3 download (stub)")
        return False
    except Exception as exc:
        logger.warning("S3 download failed: %s", exc)
        return False
