#!/usr/bin/env python3
"""One-command sync: export history and push it to a Palantir Foundry dataset.

Composes foundry_export (history -> CSV/JSONL) with foundry_client
(transactional dataset upload). Without full Foundry configuration it
falls back to exporting locally and reports what it would have done.

Usage:
    python scripts/foundry_sync.py                 # csv, upload if configured
    python scripts/foundry_sync.py --format jsonl
    python scripts/foundry_sync.py --export-only   # never upload
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from config.constants import FOUNDRY_DATASET_FILENAME
from config.settings import Settings
from scripts.foundry_client import client_from_settings
from scripts.foundry_export import build_run_rows, rows_to_csv, rows_to_jsonl

log = logging.getLogger(__name__)


def sync(
    fmt: str = "csv",
    export_only: bool = False,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Export history rows and upload them to Foundry when configured.

    Returns a summary dict: rows exported, whether an upload happened,
    and the transaction RID when one was committed.
    """
    s = settings or Settings()
    rows = build_run_rows()
    payload = rows_to_csv(rows) if fmt == "csv" else rows_to_jsonl(rows)
    suffix = ".csv" if fmt == "csv" else ".jsonl"
    target_name = FOUNDRY_DATASET_FILENAME.replace(".csv", suffix)

    summary: dict[str, object] = {
        "rows": len(rows),
        "format": fmt,
        "uploaded": False,
        "transaction_rid": None,
    }

    if export_only or not s.foundry_configured():
        if not export_only:
            log.info("Foundry not configured; skipping upload (%d rows exported)", len(rows))
        return summary

    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        client = client_from_settings(s)
        txn = client.upload_dataset_file(
            s.foundry_dataset_rid, tmp_path, target_name=target_name, branch=s.foundry_branch
        )
        summary["uploaded"] = True
        summary["transaction_rid"] = txn
        log.info("Uploaded %d rows to %s (txn %s)", len(rows), s.foundry_dataset_rid, txn)
    finally:
        tmp_path.unlink(missing_ok=True)
    return summary


def main() -> int:
    """CLI entry point for the Foundry sync."""
    parser = argparse.ArgumentParser(description="Sync history to Palantir Foundry")
    parser.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    parser.add_argument("--export-only", action="store_true", help="Export locally, never upload")
    parser.add_argument("--output", "-o", help="Also write the export to this file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summary = sync(fmt=args.format, export_only=args.export_only)

    if args.output:
        rows = build_run_rows()
        payload = rows_to_csv(rows) if args.format == "csv" else rows_to_jsonl(rows)
        Path(args.output).write_text(payload)
        log.info("Wrote export to %s", args.output)

    log.info(
        "%d rows exported (%s), uploaded=%s",
        summary["rows"],
        summary["format"],
        summary["uploaded"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
