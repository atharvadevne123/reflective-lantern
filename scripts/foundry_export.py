"""Build and serialise history run rows for Palantir Foundry ingestion."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any

from config.constants import HISTORY_DIR

logger = logging.getLogger(__name__)

DATASET_COLUMNS: list[str] = [
    "run_key",
    "repo",
    "date",
    "mode",
    "commits",
    "improvement_count",
    "tests_passed",
    "email_status",
]

NON_RECORD_FILES = {"schema.json", "commit_schedule.json", "email_status.json", "pending"}


def _normalize_mode(value: Any) -> str:
    """Return the lowercase mode string, or "" for non-string values."""
    if not isinstance(value, str):
        return ""
    return value.lower()


def _load_history_file(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON history file; returns [] on any error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data
    except Exception as exc:
        logger.warning("Skipping invalid history file %s: %s", path, exc)
        return []


def build_run_rows(history_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read all history JSON files and flatten into a list of run-level rows.

    Each element of a JSON array becomes one row with:
    - run_key  = "{repo}:{date}"
    - repo     = stem of the JSON file
    - date     = entry["date"] if present, else entry["last_run"]
    - mode     = entry["mode"] normalised to lowercase
    - commits  = entry["commits"]
    - improvement_count = len(entry["improvements"]) when present
    - tests_passed = entry["tests_passed"]
    - email_status = entry["email_status"] (string only; "" when not a string)

    Args:
        history_dir: Path to the directory containing *.json history files.
                     Defaults to the project HISTORY_DIR constant.

    Returns:
        List of row dicts sorted by (repo, date).
    """
    if history_dir is None:
        history_dir = HISTORY_DIR
    rows: list[dict[str, Any]] = []
    for path in sorted(history_dir.glob("*.json")):
        if path.name in NON_RECORD_FILES:
            continue
        entries = _load_history_file(path)
        if not entries:
            continue
        repo = path.stem
        for entry in entries:
            date = entry.get("date") or entry.get("last_run", "")
            mode = _normalize_mode(entry.get("mode", "improvement"))
            commits = int(entry.get("commits", 0))
            improvements = entry.get("improvements", [])
            improvement_count = len(improvements) if isinstance(improvements, list) else 0
            tests_passed = entry.get("tests_passed", False)
            raw_email = entry.get("email_status", "")
            email_status = raw_email if isinstance(raw_email, str) else ""
            rows.append(
                {
                    "run_key": f"{repo}:{date}",
                    "repo": repo,
                    "date": date,
                    "mode": mode,
                    "commits": commits,
                    "improvement_count": improvement_count,
                    "tests_passed": tests_passed,
                    "email_status": email_status,
                }
            )
    rows.sort(key=lambda r: (r["repo"], r["date"]))
    return rows


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    """Serialise rows to a CSV string with header matching DATASET_COLUMNS.

    Args:
        rows: List of run-level row dicts.

    Returns:
        CSV text with a header row and one data row per element.
    """
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=DATASET_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def rows_to_jsonl(rows: list[dict[str, Any]]) -> str:
    """Serialise rows to a JSONL string (one JSON object per line).

    Args:
        rows: List of run-level row dicts.

    Returns:
        JSONL text with one JSON object per line, or "" for empty input.
    """
    if not rows:
        return ""
    return "\n".join(json.dumps(row) for row in rows)


def build_ontology_objects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert run rows into Foundry ontology-compatible object dicts.

    Returns two object types:
    - ``Repository``: one per unique repo, with aggregated totalRuns/totalCommits.
    - ``AutomationRun``: one per row, with run-level properties.

    Args:
        rows: List of run-level row dicts from :func:`build_run_rows`.

    Returns:
        List of ontology objects suitable for Foundry Ontology SDK upload.
        Empty list when rows is empty.
    """
    if not rows:
        return []

    repo_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        repo = row["repo"]
        if repo not in repo_stats:
            repo_stats[repo] = {"totalRuns": 0, "totalCommits": 0}
        repo_stats[repo]["totalRuns"] += 1
        repo_stats[repo]["totalCommits"] += int(row.get("commits", 0))

    objects: list[dict[str, Any]] = []

    for repo, stats in repo_stats.items():
        objects.append(
            {
                "objectType": "Repository",
                "primaryKey": repo,
                "properties": {
                    "totalRuns": stats["totalRuns"],
                    "totalCommits": stats["totalCommits"],
                },
            }
        )

    for row in rows:
        objects.append(
            {
                "objectType": "AutomationRun",
                "primaryKey": row["run_key"],
                "properties": {
                    "repository": row["repo"],
                    "runDate": row["date"],
                    "mode": row["mode"],
                    "commits": row["commits"],
                    "improvementCount": row["improvement_count"],
                    "testsPassed": row["tests_passed"],
                    "emailStatus": row["email_status"],
                },
            }
        )

    return objects


__all__ = ["build_run_rows", "rows_to_jsonl", "build_ontology_objects"]
