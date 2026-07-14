"""Palantir Foundry API client for dataset upload."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FoundryConfigError(RuntimeError):
    """Raised when Foundry credentials are missing or invalid."""


class FoundryAPIError(RuntimeError):
    """Raised when the Foundry API returns a non-2xx response."""


class FoundryClient:
    """Thin HTTP client for the Palantir Foundry Dataset API.

    Args:
        hostname: Base URL of the Foundry stack (e.g. https://stack.palantirfoundry.com).
        token: Bearer auth token.
    """

    def __init__(self, hostname: str, token: str) -> None:
        if not hostname:
            raise FoundryConfigError("Foundry hostname must not be empty.")
        if not token:
            raise FoundryConfigError("Foundry token must not be empty.")
        self.base_url = hostname.rstrip("/") + "/api/v2"
        self._token = token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make an authenticated JSON request and return the parsed body."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise FoundryAPIError(f"Foundry API error {exc.code}: {exc.reason}") from exc
        except FoundryAPIError:
            raise
        except Exception as exc:
            raise FoundryAPIError(f"Foundry request failed: {exc}") from exc

    def _raw_request(self, method: str, url: str) -> bytes:
        """Make an authenticated request and return raw response bytes."""
        req = urllib.request.Request(url, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise FoundryAPIError(f"Foundry API error {exc.code}: {exc.reason}") from exc
        except Exception as exc:
            raise FoundryAPIError(f"Foundry request failed: {exc}") from exc

    def create_transaction(self, dataset_rid: str, branch: str = "master") -> str:
        """Open a new write transaction on a Foundry dataset.

        Args:
            dataset_rid: The dataset resource identifier.
            branch: Branch name (default "master").

        Returns:
            The new transaction RID string.

        Raises:
            FoundryAPIError: If the response contains no transaction RID.
        """
        url = f"{self.base_url}/datasets/{dataset_rid}/transactions?branchName={branch}"
        result = self._request("POST", url, {"transactionType": "SNAPSHOT"})
        if "rid" not in result:
            raise FoundryAPIError(f"No transaction RID in response: {result}")
        return result["rid"]

    def abort_transaction(self, dataset_rid: str, transaction_rid: str) -> None:
        """Abort an open write transaction without committing.

        Args:
            dataset_rid: The dataset resource identifier.
            transaction_rid: Open transaction RID to abort.
        """
        url = f"{self.base_url}/datasets/{dataset_rid}/transactions/{transaction_rid}/abort"
        try:
            self._request("POST", url)
        except Exception as exc:
            logger.warning("Failed to abort transaction %s: %s", transaction_rid, exc)

    def upload_file(
        self, dataset_rid: str, transaction_rid: str, filename: str, content: bytes
    ) -> None:
        """Upload a file into an open transaction.

        Args:
            dataset_rid: The dataset resource identifier.
            transaction_rid: Open transaction RID.
            filename: Logical filename within the dataset.
            content: Raw bytes to upload.
        """
        url = (
            f"{self.base_url}/datasets/{dataset_rid}/files/{filename}"
            f"/upload?transactionRid={transaction_rid}"
        )
        req = urllib.request.Request(
            url,
            data=content,
            headers={**self._headers(), "Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60):
                pass
        except urllib.error.HTTPError as exc:
            raise FoundryAPIError(f"File upload error {exc.code}: {exc.reason}") from exc

    def commit_transaction(self, dataset_rid: str, transaction_rid: str) -> None:
        """Commit and close an open write transaction.

        Args:
            dataset_rid: The dataset resource identifier.
            transaction_rid: Open transaction RID to commit.
        """
        url = f"{self.base_url}/datasets/{dataset_rid}/transactions/{transaction_rid}/commit"
        self._request("POST", url)

    def upload_dataset_file(
        self,
        dataset_rid: str,
        local_path: str | Path,
        target_name: str | None = None,
        branch: str = "master",
    ) -> str:
        """Upload a single local file in a self-contained transaction.

        Args:
            dataset_rid: The dataset resource identifier.
            local_path: Path to the local file to upload.
            target_name: Logical name within the dataset (defaults to filename).
            branch: Branch name for the transaction.

        Returns:
            The committed transaction RID.

        Raises:
            FoundryAPIError: If upload fails (transaction is aborted first).
        """
        path = Path(local_path)
        name = target_name or path.name
        transaction_rid = self.create_transaction(dataset_rid, branch=branch)
        try:
            content = path.read_bytes()
            self.upload_file(dataset_rid, transaction_rid, name, content)
        except Exception as exc:
            self.abort_transaction(dataset_rid, transaction_rid)
            raise FoundryAPIError(f"Upload of {name} failed: {exc}") from exc
        self.commit_transaction(dataset_rid, transaction_rid)
        return transaction_rid

    def upload_dataset_files(
        self,
        dataset_rid: str,
        files: dict[str, bytes | str],
        branch: str = "master",
    ) -> str:
        """Upload multiple files in a single transaction.

        Args:
            dataset_rid: The dataset resource identifier.
            files: Mapping of logical filename to content (bytes or str).
            branch: Branch name for the transaction.

        Returns:
            The committed transaction RID.

        Raises:
            ValueError: If files dict is empty.
            FoundryAPIError: If any upload fails (transaction is aborted first).
        """
        if not files:
            raise ValueError("files dict must not be empty")
        transaction_rid = self.create_transaction(dataset_rid, branch=branch)
        try:
            for filename, content in files.items():
                data = content.encode() if isinstance(content, str) else content
                self.upload_file(dataset_rid, transaction_rid, filename, data)
        except Exception as exc:
            self.abort_transaction(dataset_rid, transaction_rid)
            raise FoundryAPIError(f"Multi-file upload failed: {exc}") from exc
        self.commit_transaction(dataset_rid, transaction_rid)
        return transaction_rid

    def get_dataset(self, dataset_rid: str) -> dict[str, Any]:
        """Return metadata for a Foundry dataset.

        Args:
            dataset_rid: The dataset resource identifier.

        Returns:
            Dict of dataset metadata.
        """
        url = f"{self.base_url}/datasets/{dataset_rid}"
        return self._request("GET", url)

    def list_files(self, dataset_rid: str = "", branch: str = "master") -> list[str]:
        """List files in a Foundry dataset branch.

        Args:
            dataset_rid: The dataset resource identifier.
            branch: Branch name.

        Returns:
            List of logical file paths, or [] on error/missing dataset.
        """
        if not dataset_rid:
            return []
        url = f"{self.base_url}/datasets/{dataset_rid}/files/list?branchName={branch}"
        try:
            result = self._request("GET", url)
            data = result.get("data", [])
            if not isinstance(data, list):
                return []
            return [item["path"] for item in data if isinstance(item, dict) and "path" in item]
        except Exception:
            return []

    def read_table(
        self, dataset_rid: str, branch: str = "master", format: str = "CSV"
    ) -> bytes:
        """Read a Foundry dataset as raw bytes.

        Args:
            dataset_rid: The dataset resource identifier.
            branch: Branch name.
            format: Output format (default "CSV").

        Returns:
            Raw response bytes.
        """
        url = f"{self.base_url}/datasets/{dataset_rid}/readTable?format={format}&branchName={branch}"
        return self._raw_request("GET", url)


def client_from_settings(settings: Any = None) -> FoundryClient:
    """Create a FoundryClient from a Settings object (or the module singleton).

    Args:
        settings: Application settings. Uses module singleton when None.

    Returns:
        A configured FoundryClient.

    Raises:
        FoundryConfigError: If hostname or token is not set.
    """
    if settings is None:
        from config.settings import Settings
        settings = Settings()
    return FoundryClient(settings.foundry_hostname, settings.foundry_token)


def main() -> int:
    """CLI entry point for Foundry operations."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Foundry dataset operations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true", help="Verify Foundry connectivity")
    group.add_argument("--upload", metavar="FILE", help="Upload a file to Foundry")
    args = parser.parse_args()

    from config.settings import Settings
    settings = Settings()

    if args.verify:
        if not settings.foundry_configured():
            print("Foundry is not configured (missing hostname/token/dataset_rid).", file=sys.stderr)
            return 1
        try:
            client = client_from_settings(settings)
            client.get_dataset(settings.foundry_dataset_rid)
            print("Foundry connection verified.")
            return 0
        except Exception as exc:
            print(f"Foundry verification failed: {exc}", file=sys.stderr)
            return 1

    if args.upload:
        if not settings.foundry_configured():
            print("Foundry is not configured.", file=sys.stderr)
            return 1
        try:
            client = client_from_settings(settings)
            txn = client.upload_dataset_file(settings.foundry_dataset_rid, args.upload)
            print(f"Uploaded successfully. Transaction: {txn}")
            return 0
        except Exception as exc:
            print(f"Upload failed: {exc}", file=sys.stderr)
            return 1

    return 0
