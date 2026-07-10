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
import json
import logging
import sys
from pathlib import Path

from config.constants import FOUNDRY_DATASET_FILENAME, VERSION
from config.settings import Settings
from scripts.foundry_client import client_from_settings
from scripts.foundry_export import (
    build_ontology_objects,
    build_run_rows,
    rows_to_csv,
    rows_to_jsonl,
)

log = logging.getLogger(__name__)

ONTOLOGY_FILENAME = "reflective_lantern_ontology.json"
MANIFEST_FILENAME = "manifest.json"


def build_manifest(rows: int, fmt: str, include_ontology: bool) -> dict[str, object]:
    """Describe the contents of one sync transaction."""
    return {
        "generator": "reflective-lantern",
        "version": VERSION,
        "rows": rows,
        "format": fmt,
        "includes_ontology": include_ontology,
        "columns": "run_key,repo,date,mode,commits,tests_passed,improvement_count,email_status",
    }


def sync(
    fmt: str = "csv",
    export_only: bool = False,
    include_ontology: bool = True,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Export history rows and upload them to Foundry when configured.

    Uploads the tabular export, an ontology-object JSON, and a manifest in
    one atomic transaction, then verifies the files are visible. Returns a
    summary dict: rows exported, whether an upload happened, the transaction
    RID, and the verified file list.
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
        "files": [],
    }

    if export_only or not s.foundry_configured():
        if not export_only:
            log.info("Foundry not configured; skipping upload (%d rows exported)", len(rows))
        return summary

    files: dict[str, bytes] = {target_name: payload.encode()}
    if include_ontology:
        files[ONTOLOGY_FILENAME] = json.dumps(build_ontology_objects(rows), indent=2).encode()
    files[MANIFEST_FILENAME] = json.dumps(
        build_manifest(len(rows), fmt, include_ontology), indent=2
    ).encode()

    client = client_from_settings(s)
    txn = client.upload_dataset_files(s.foundry_dataset_rid, files, branch=s.foundry_branch)
    summary["uploaded"] = True
    summary["transaction_rid"] = txn

    visible = client.list_files(s.foundry_dataset_rid, branch=s.foundry_branch)
    summary["files"] = visible
    missing = sorted(set(files) - set(visible))
    if missing:
        log.warning("Committed txn %s but files not yet visible: %s", txn, missing)
    log.info(
        "Uploaded %d rows + %d file(s) to %s (txn %s)",
        len(rows),
        len(files),
        s.foundry_dataset_rid,
        txn,
    )
    return summary


def main() -> int:
    """CLI entry point for the Foundry sync."""
    parser = argparse.ArgumentParser(description="Sync history to Palantir Foundry")
    parser.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    parser.add_argument("--export-only", action="store_true", help="Export locally, never upload")
    parser.add_argument(
        "--no-ontology",
        action="store_true",
        help="Skip the ontology-object JSON upload",
    )
    parser.add_argument("--output", "-o", help="Also write the export to this file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summary = sync(
        fmt=args.format,
        export_only=args.export_only,
        include_ontology=not args.no_ontology,
    )

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
