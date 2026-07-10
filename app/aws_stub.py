"""AWS/boto3 stub for S3 model artifact storage."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

AWS_ENABLED = os.getenv("AWS_ENABLED", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "volt-cast-models")
S3_PREFIX = os.getenv("S3_PREFIX", "models/")


def upload_model(local_path: str, s3_key: str | None = None) -> str:
    """Upload model file to S3 — no-op stub when AWS_ENABLED=false."""
    path = Path(local_path)
    key = s3_key or f"{S3_PREFIX}{path.name}"

    if not AWS_ENABLED:
        logger.info("aws stub — would upload %s to s3://%s/%s", local_path, S3_BUCKET, key)
        return f"s3://{S3_BUCKET}/{key} (stub)"

    try:
        import boto3  # type: ignore[import]
        s3 = boto3.client("s3")
        s3.upload_file(local_path, S3_BUCKET, key)
        uri = f"s3://{S3_BUCKET}/{key}"
        logger.info("uploaded model to %s", uri)
        return uri
    except Exception as exc:
        logger.error("s3 upload failed: %s", exc)
        raise


def download_model(s3_key: str, local_path: str) -> str:
    """Download model file from S3 — no-op stub when AWS_ENABLED=false."""
    if not AWS_ENABLED:
        logger.info("aws stub — would download s3://%s/%s to %s", S3_BUCKET, s3_key, local_path)
        return local_path

    try:
        import boto3  # type: ignore[import]
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, s3_key, local_path)
        logger.info("downloaded model from s3://%s/%s", S3_BUCKET, s3_key)
        return local_path
    except Exception as exc:
        logger.error("s3 download failed: %s", exc)
        raise


def list_model_versions() -> list[str]:
    """List available model versions in S3."""
    if not AWS_ENABLED:
        logger.info("aws stub — would list versions in s3://%s/%s", S3_BUCKET, S3_PREFIX)
        return ["stub/volt_cast_model_v1.0.0.joblib"]

    try:
        import boto3  # type: ignore[import]
        s3 = boto3.client("s3")
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception as exc:
        logger.error("s3 list failed: %s", exc)
        return []
