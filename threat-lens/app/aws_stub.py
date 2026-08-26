"""S3 model artefact storage with an in-memory stub fallback.

Uses boto3 when it is installed and `S3_BUCKET` is configured; otherwise keeps
artefacts in a process-local dict so the code path stays exercisable in tests
and offline environments.
"""

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HAS_BOTO3 = importlib.util.find_spec("boto3") is not None


class ModelArtifactStore:
    """Uploads and downloads model artefacts, backed by S3 or memory."""

    def __init__(self, bucket: str | None = None, prefix: str = "threat-lens") -> None:
        self.bucket = bucket or os.getenv("S3_BUCKET")
        self.prefix = prefix
        self.enabled = bool(_HAS_BOTO3 and self.bucket)
        self._memory: dict[str, bytes] = {}
        if not self.enabled:
            logger.info("S3 unavailable; artefacts will be held in memory only")

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def upload(self, local_path: Path, name: str | None = None) -> str:
        """Upload a local file and return the storage key.

        Args:
            local_path: File to upload.
            name: Object name; defaults to the file's own name.

        Returns:
            The `s3://` URI when S3-backed, otherwise a `memory://` URI.
        """
        if not local_path.exists():
            raise FileNotFoundError(f"No artefact at {local_path}")

        key = self._key(name or local_path.name)
        if self.enabled:
            try:
                import boto3  # noqa: PLC0415

                boto3.client("s3").upload_file(str(local_path), self.bucket, key)
                logger.info("Uploaded %s to s3://%s/%s", local_path, self.bucket, key)
                return f"s3://{self.bucket}/{key}"
            except Exception:
                logger.exception("S3 upload failed; holding artefact in memory")

        self._memory[key] = local_path.read_bytes()
        return f"memory://{key}"

    def download(self, name: str, dest: Path) -> Path:
        """Fetch an artefact into `dest` and return that path."""
        key = self._key(name)
        if self.enabled:
            try:
                import boto3  # noqa: PLC0415

                boto3.client("s3").download_file(self.bucket, key, str(dest))
                logger.info("Downloaded s3://%s/%s to %s", self.bucket, key, dest)
                return dest
            except Exception:
                logger.exception("S3 download failed; trying in-memory store")

        if key not in self._memory:
            raise KeyError(f"No artefact stored under {key}")
        dest.write_bytes(self._memory[key])
        return dest

    def exists(self, name: str) -> bool:
        """Report whether an artefact is present in the active backend."""
        key = self._key(name)
        if self.enabled:
            try:
                import boto3  # noqa: PLC0415

                boto3.client("s3").head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception:
                return False
        return key in self._memory

    def list_artifacts(self) -> list[str]:
        """List artefact keys held by the active backend."""
        if self.enabled:
            try:
                import boto3  # noqa: PLC0415

                resp: dict[str, Any] = boto3.client("s3").list_objects_v2(
                    Bucket=self.bucket, Prefix=self.prefix
                )
                return [o["Key"] for o in resp.get("Contents", [])]
            except Exception:
                logger.exception("S3 listing failed; returning in-memory keys")
        return sorted(self._memory)
