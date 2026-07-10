#!/usr/bin/env python3
"""Minimal Palantir Foundry Datasets API client (v2, transaction-based).

Implements the standard Foundry file-upload flow against the REST API:
create a SNAPSHOT transaction, upload one or more files into it, then
commit (or abort on failure). Uses only the standard library.

Configuration comes from the environment (see config.settings.Settings):
    FOUNDRY_HOSTNAME     e.g. https://mystack.palantirfoundry.com
    FOUNDRY_TOKEN        bearer token
    FOUNDRY_DATASET_RID  e.g. ri.foundry.main.dataset.xxxx
    FOUNDRY_BRANCH       defaults to master

Usage:
    python scripts/foundry_client.py --upload runs.csv
    python scripts/foundry_client.py --upload runs.csv --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from config.constants import (
    FOUNDRY_API_VERSION,
    FOUNDRY_DATASET_FILENAME,
    FOUNDRY_UPLOAD_TIMEOUT_SECONDS,
)
from config.settings import Settings

log = logging.getLogger(__name__)


class FoundryConfigError(RuntimeError):
    """Raised when required Foundry configuration is missing."""


class FoundryAPIError(RuntimeError):
    """Raised when the Foundry API returns an error response."""


class FoundryClient:
    """Thin client for the Foundry Datasets v2 REST API."""

    def __init__(self, hostname: str, token: str) -> None:
        if not hostname or not token:
            raise FoundryConfigError("FOUNDRY_HOSTNAME and FOUNDRY_TOKEN must both be set")
        self.base_url = f"{hostname.rstrip('/')}/api/{FOUNDRY_API_VERSION}"
        self._token = token

    def _request_raw(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> bytes:
        """Issue an authenticated request and return the raw response body."""
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": content_type,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=FOUNDRY_UPLOAD_TIMEOUT_SECONDS) as resp:
                return resp.read()  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise FoundryAPIError(f"{method} {path} failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise FoundryAPIError(f"{method} {path} failed: {exc.reason}") from exc

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        """Issue an authenticated request and return the parsed JSON body."""
        raw = self._request_raw(method, path, body=body, content_type=content_type)
        if not raw:
            return {}
        result = json.loads(raw)
        return result if isinstance(result, dict) else {"data": result}

    def create_transaction(
        self, dataset_rid: str, branch: str = "master", txn_type: str = "SNAPSHOT"
    ) -> str:
        """Open a transaction on *dataset_rid* and return its RID."""
        body = json.dumps({"transactionType": txn_type}).encode()
        resp = self._request(
            "POST",
            f"/datasets/{dataset_rid}/transactions?branchName={branch}",
            body=body,
        )
        rid = resp.get("rid")
        if not isinstance(rid, str):
            raise FoundryAPIError(f"No transaction RID in response: {resp}")
        return rid

    def upload_file(
        self, dataset_rid: str, transaction_rid: str, file_path: str, content: bytes
    ) -> None:
        """Upload *content* as *file_path* inside an open transaction."""
        self._request(
            "POST",
            f"/datasets/{dataset_rid}/files/{file_path}/upload?transactionRid={transaction_rid}",
            body=content,
            content_type="application/octet-stream",
        )

    def commit_transaction(self, dataset_rid: str, transaction_rid: str) -> None:
        """Commit an open transaction, making its files visible."""
        self._request("POST", f"/datasets/{dataset_rid}/transactions/{transaction_rid}/commit")

    def abort_transaction(self, dataset_rid: str, transaction_rid: str) -> None:
        """Abort an open transaction, discarding its uploads."""
        self._request("POST", f"/datasets/{dataset_rid}/transactions/{transaction_rid}/abort")

    def get_dataset(self, dataset_rid: str) -> dict[str, Any]:
        """Fetch dataset metadata. Useful to verify the RID and token work."""
        return self._request("GET", f"/datasets/{dataset_rid}")

    def list_files(self, dataset_rid: str, branch: str = "master") -> list[str]:
        """Return the file paths currently in *dataset_rid* on *branch*."""
        resp = self._request("GET", f"/datasets/{dataset_rid}/files?branchName={branch}")
        data = resp.get("data", [])
        if not isinstance(data, list):
            return []
        return [f["path"] for f in data if isinstance(f, dict) and "path" in f]

    def read_table(self, dataset_rid: str, fmt: str = "CSV", branch: str = "master") -> bytes:
        """Read the dataset back as a table export (requires a schema).

        *fmt* is one of Foundry's readTable formats, e.g. CSV or ARROW.
        """
        return self._request_raw(
            "GET",
            f"/datasets/{dataset_rid}/readTable?format={fmt}&branchName={branch}",
        )

    def upload_dataset_files(
        self,
        dataset_rid: str,
        files: dict[str, bytes],
        branch: str = "master",
    ) -> str:
        """Upload several files to *dataset_rid* in one atomic transaction.

        *files* maps target path inside the dataset to content bytes.
        Returns the committed transaction RID. Aborts the transaction if
        any upload or the commit fails, then re-raises.
        """
        if not files:
            raise ValueError("files must not be empty")
        txn = self.create_transaction(dataset_rid, branch=branch)
        try:
            for target_name, content in files.items():
                self.upload_file(dataset_rid, txn, target_name, content)
            self.commit_transaction(dataset_rid, txn)
        except FoundryAPIError:
            log.warning("Upload failed; aborting transaction %s", txn)
            try:
                self.abort_transaction(dataset_rid, txn)
            except FoundryAPIError:
                log.error("Could not abort transaction %s", txn)
            raise
        return txn

    def upload_dataset_file(
        self,
        dataset_rid: str,
        local_path: Path,
        target_name: str = FOUNDRY_DATASET_FILENAME,
        branch: str = "master",
    ) -> str:
        """Upload one local file to *dataset_rid* in a single transaction.

        Returns the committed transaction RID. Aborts the transaction if
        the upload or commit fails, then re-raises.
        """
        return self.upload_dataset_files(
            dataset_rid, {target_name: local_path.read_bytes()}, branch=branch
        )


def client_from_settings(settings: Settings | None = None) -> FoundryClient:
    """Build a FoundryClient from environment-backed Settings."""
    s = settings or Settings()
    return FoundryClient(s.foundry_hostname, s.foundry_token)


def main() -> int:
    """Upload a file to the configured Foundry dataset."""
    parser = argparse.ArgumentParser(description="Upload files to a Foundry dataset")
    parser.add_argument("--upload", help="Local file to upload")
    parser.add_argument("--name", help="Target filename inside the dataset")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and file, do not upload"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check that the configured dataset is reachable, then exit",
    )
    parser.add_argument(
        "--read-table",
        action="store_true",
        help="Read the dataset back as CSV and print it (requires an applied schema)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.upload and not args.verify and not args.read_table:
        parser.error("one of --upload, --verify, or --read-table is required")

    settings = Settings()
    if not settings.foundry_configured():
        log.error(
            "Foundry is not configured — set FOUNDRY_HOSTNAME, FOUNDRY_TOKEN, "
            "and FOUNDRY_DATASET_RID"
        )
        return 1

    if args.verify:
        client = client_from_settings(settings)
        try:
            meta = client.get_dataset(settings.foundry_dataset_rid)
            files = client.list_files(settings.foundry_dataset_rid, branch=settings.foundry_branch)
        except FoundryAPIError as exc:
            log.error("Verification failed: %s", exc)
            log.error(
                "Config looks complete but the request did not succeed — check the "
                "network policy allows %s and that the token is valid",
                settings.foundry_hostname,
            )
            return 1
        log.info(
            "Dataset reachable: %s (%d file(s) on branch %s)",
            meta.get("name", settings.foundry_dataset_rid),
            len(files),
            settings.foundry_branch,
        )
        return 0

    if args.read_table:
        client = client_from_settings(settings)
        try:
            data = client.read_table(settings.foundry_dataset_rid, branch=settings.foundry_branch)
        except FoundryAPIError as exc:
            log.error("readTable failed: %s", exc)
            log.error("Hint: readTable requires a schema applied to the dataset in Foundry")
            return 1
        sys.stdout.write(data.decode(errors="replace"))
        return 0

    local = Path(args.upload)
    if not local.is_file():
        log.error("File not found: %s", local)
        return 1

    target = args.name or local.name
    if args.dry_run:
        log.info(
            "[dry-run] Would upload %s (%d bytes) as '%s' to dataset %s on branch %s",
            local,
            local.stat().st_size,
            target,
            settings.foundry_dataset_rid,
            settings.foundry_branch,
        )
        return 0

    client = client_from_settings(settings)
    try:
        txn = client.upload_dataset_file(
            settings.foundry_dataset_rid, local, target_name=target, branch=settings.foundry_branch
        )
    except FoundryAPIError as exc:
        log.error("Upload failed: %s", exc)
        return 1
    log.info("Committed transaction %s", txn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
