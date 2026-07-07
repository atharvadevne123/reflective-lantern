#!/usr/bin/env python3
"""Export Reflective Lantern history as Palantir Foundry-ready datasets.

Flattens per-repo history JSON into a tabular dataset (CSV or JSONL) with a
stable schema suitable for upload to a Foundry dataset, and can also emit
Foundry Ontology object payloads (AutomationRun / Repository object types).

Usage:
    python scripts/foundry_export.py --format csv --output runs.csv
    python scripts/foundry_export.py --format jsonl
    python scripts/foundry_export.py --format ontology
    python scripts/foundry_export.py --repo reflective-lantern --format csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

from config.constants import (
    FOUNDRY_REPO_OBJECT_TYPE,
    FOUNDRY_RUN_OBJECT_TYPE,
    NON_RECORD_FILES,
)

log = logging.getLogger(__name__)
HISTORY_DIR = Path(__file__).parent.parent / "history"

# Stable column order for the Foundry dataset schema
DATASET_COLUMNS: tuple[str, ...] = (
    "run_key",
    "repo",
    "date",
    "mode",
    "commits",
    "tests_passed",
    "improvement_count",
    "email_status",
)


def _normalize_mode(raw: object) -> str:
    """Return the run mode lowercased, or empty string when absent."""
    return str(raw).lower() if isinstance(raw, str) else ""


def build_run_rows(history_dir: Path = HISTORY_DIR) -> list[dict[str, Any]]:
    """Flatten every history entry into one dataset row per run.

    Rows are sorted by (repo, date) and keyed by ``run_key`` = "<repo>:<date>"
    so re-exports upsert cleanly in Foundry.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(history_dir.glob("*.json")):
        if path.name in NON_RECORD_FILES:
            continue
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("Skipping invalid JSON file: %s", path.name)
            continue
        entries = raw if isinstance(raw, list) else [raw]
        repo = path.stem
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            run_date = str(entry.get("date") or entry.get("last_run") or "")
            improvements = entry.get("improvements")
            email_status = entry.get("email_status")
            rows.append(
                {
                    "run_key": f"{repo}:{run_date}",
                    "repo": repo,
                    "date": run_date,
                    "mode": _normalize_mode(entry.get("mode")),
                    "commits": entry.get("commits", 0),
                    "tests_passed": bool(entry.get("tests_passed", False)),
                    "improvement_count": len(improvements) if isinstance(improvements, list) else 0,
                    "email_status": email_status if isinstance(email_status, str) else "",
                }
            )
    rows.sort(key=lambda r: (r["repo"], r["date"]))
    return rows


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    """Render *rows* as CSV text with the stable DATASET_COLUMNS header."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=DATASET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def rows_to_jsonl(rows: list[dict[str, Any]]) -> str:
    """Render *rows* as newline-delimited JSON (one object per line)."""
    return "\n".join(json.dumps(r, sort_keys=True) for r in rows) + ("\n" if rows else "")


def build_ontology_objects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map dataset rows to Foundry Ontology object payloads.

    Emits one Repository object per distinct repo and one AutomationRun
    object per row, shaped like Foundry's ontology object API payloads.
    """
    objects: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    for row in rows:
        repo = str(row["repo"])
        if repo not in seen_repos:
            seen_repos.add(repo)
            repo_rows = [r for r in rows if r["repo"] == repo]
            objects.append(
                {
                    "objectType": FOUNDRY_REPO_OBJECT_TYPE,
                    "primaryKey": repo,
                    "properties": {
                        "name": repo,
                        "totalRuns": len(repo_rows),
                        "totalCommits": sum(int(r["commits"]) for r in repo_rows),
                    },
                }
            )
        objects.append(
            {
                "objectType": FOUNDRY_RUN_OBJECT_TYPE,
                "primaryKey": row["run_key"],
                "properties": {
                    "repository": repo,
                    "runDate": row["date"],
                    "mode": row["mode"],
                    "commits": row["commits"],
                    "testsPassed": row["tests_passed"],
                    "improvementCount": row["improvement_count"],
                    "emailStatus": row["email_status"],
                },
            }
        )
    return objects


def main() -> int:
    """Export history to the chosen Foundry-ready format."""
    parser = argparse.ArgumentParser(description="Export history for Palantir Foundry")
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl", "ontology"),
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument("--output", "-o", help="Write to file instead of stdout")
    parser.add_argument("--repo", help="Only export rows for this repository")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = build_run_rows()
    if args.repo:
        rows = [r for r in rows if r["repo"] == args.repo]
        if not rows:
            log.error("No history rows found for repo '%s'", args.repo)
            return 1

    if args.format == "csv":
        payload = rows_to_csv(rows)
    elif args.format == "jsonl":
        payload = rows_to_jsonl(rows)
    else:
        payload = json.dumps(build_ontology_objects(rows), indent=2)

    if args.output:
        Path(args.output).write_text(payload)
        log.info("Wrote %d rows to %s", len(rows), args.output)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
