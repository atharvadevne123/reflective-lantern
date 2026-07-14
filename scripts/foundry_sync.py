"""Orchestrate a Foundry dataset sync from history files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config.constants import HISTORY_DIR
from config.settings import Settings, get_settings
from scripts.foundry_client import FoundryConfigError, client_from_settings
from scripts.foundry_export import build_ontology_objects, build_run_rows, rows_to_csv, rows_to_jsonl

logger = logging.getLogger(__name__)

_CSV_FILENAME = "reflective_lantern_runs.csv"
_JSONL_FILENAME = "reflective_lantern_runs.jsonl"
_ONTOLOGY_FILENAME = "reflective_lantern_ontology.json"
_MANIFEST_FILENAME = "manifest.json"


def build_manifest(rows: int, fmt: str, include_ontology: bool) -> dict[str, Any]:
    """Build a manifest dict describing the exported dataset."""
    return {
        "rows": rows,
        "format": fmt,
        "includes_ontology": include_ontology,
        "generator": "reflective-lantern",
    }


def sync(
    *,
    settings: Settings | None = None,
    fmt: str = "csv",
    export_only: bool = False,
    include_ontology: bool = True,
    history_dir: Path | None = None,
) -> dict[str, Any]:
    """Build history rows and optionally upload them to a Foundry dataset.

    Args:
        settings: Application settings. Uses module singleton when None.
        fmt: Output format - "csv" or "jsonl".
        export_only: When True, build export data but never upload.
        include_ontology: Whether to include the ontology JSON file.
        history_dir: Override for the history directory path.

    Returns:
        Summary dict with keys: rows, format, uploaded, transaction_rid, files.
    """
    if settings is None:
        settings = get_settings()
    if history_dir is None:
        history_dir = HISTORY_DIR

    rows = build_run_rows(history_dir)
    row_count = len(rows)

    if fmt == "jsonl":
        data_filename = _JSONL_FILENAME
        data_content = rows_to_jsonl(rows)
    else:
        fmt = "csv"
        data_filename = _CSV_FILENAME
        data_content = rows_to_csv(rows)

    files: dict[str, str] = {data_filename: data_content}
    if include_ontology:
        ontology_objects = build_ontology_objects(rows)
        files[_ONTOLOGY_FILENAME] = json.dumps(ontology_objects, indent=2)

    manifest = build_manifest(row_count, fmt, include_ontology)
    files[_MANIFEST_FILENAME] = json.dumps(manifest)

    summary: dict[str, Any] = {
        "rows": row_count,
        "format": fmt,
        "uploaded": False,
        "transaction_rid": None,
        "files": [],
    }

    if not export_only and settings.foundry_configured():
        try:
            client = client_from_settings(settings)
            transaction_rid = client.upload_dataset_files(settings.foundry_dataset_rid, files)
            uploaded_files = client.list_files(settings.foundry_dataset_rid)
            summary["uploaded"] = True
            summary["transaction_rid"] = transaction_rid
            summary["files"] = uploaded_files
        except (FoundryConfigError, Exception) as exc:
            logger.warning("Foundry upload failed: %s", exc)

    return summary
