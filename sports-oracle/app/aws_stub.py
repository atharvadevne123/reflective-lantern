"""AWS S3 model artifact stub for Sports-Oracle."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_model(local_path: Path, bucket: str, key: str) -> bool:
    """Upload model artifact to S3 (stub: logs intent, no actual upload)."""
    try:
        import boto3  # type: ignore[import-untyped]
        s3 = boto3.client("s3")
        s3.upload_file(str(local_path), bucket, key)
        logger.info("model_uploaded_to_s3", extra={"bucket": bucket, "key": key})
        return True
    except ImportError:
        logger.warning("boto3_not_installed_skipping_s3_upload")
        return False
    except Exception:
        logger.exception("s3_upload_failed", extra={"bucket": bucket, "key": key})
        return False


def download_model(bucket: str, key: str, local_path: Path) -> bool:
    """Download model artifact from S3 (stub)."""
    try:
        import boto3  # type: ignore[import-untyped]
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(local_path))
        logger.info("model_downloaded_from_s3", extra={"bucket": bucket, "key": key})
        return True
    except ImportError:
        logger.warning("boto3_not_installed_skipping_s3_download")
        return False
    except Exception:
        logger.exception("s3_download_failed")
        return False
