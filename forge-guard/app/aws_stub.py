"""AWS S3 model artefact upload stub for Forge-Guard.

Uploads model.joblib to S3 after training, or logs a warning when boto3
is unavailable or credentials are not configured.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "forge-guard/models")


def upload_model(model_path: Path = Path("model.joblib"), version: str = "1.0.0") -> bool:
    """Upload the serialised model to S3.

    Returns True on success, False when boto3 is missing or credentials fail.
    The stub is a no-op when S3_BUCKET is not configured so local dev is unaffected.
    """
    if not S3_BUCKET:
        logger.debug("S3_BUCKET not set — skipping model upload.")
        return False

    try:
        import boto3  # type: ignore[import]

        s3 = boto3.client("s3")
        key = f"{S3_PREFIX}/{version}/{model_path.name}"
        s3.upload_file(str(model_path), S3_BUCKET, key)
        logger.info("Model uploaded to s3://%s/%s", S3_BUCKET, key)
        return True
    except ImportError:
        logger.warning("boto3 not installed — model not uploaded to S3.")
    except Exception as exc:
        logger.warning("S3 upload failed: %s", exc)
    return False


def download_model(dest: Path = Path("model.joblib"), version: str = "1.0.0") -> bool:
    """Download the serialised model from S3."""
    if not S3_BUCKET:
        return False

    try:
        import boto3  # type: ignore[import]

        s3 = boto3.client("s3")
        key = f"{S3_PREFIX}/{version}/{dest.name}"
        s3.download_file(S3_BUCKET, key, str(dest))
        logger.info("Model downloaded from s3://%s/%s", S3_BUCKET, key)
        return True
    except ImportError:
        logger.warning("boto3 not installed.")
    except Exception as exc:
        logger.warning("S3 download failed: %s", exc)
    return False
