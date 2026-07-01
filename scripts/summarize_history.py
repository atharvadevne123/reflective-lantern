#!/usr/bin/env python3
"""Summarise all history JSON files as a human-readable table.

Usage:
    python scripts/summarize_history.py [--sort-by commits|date|repo]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
HISTORY_DIR = Path(__file__).parent.parent / "history"


def load_latest_entry(path: Path) -> dict[str, Any] | None:
    """Return the most recent entry from a history file."""
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        log.warning("Skipping %s: %s", path.name, exc)
        return None
    if isinstance(raw, list) and raw:
        # Pick the entry with the latest date
        def _date_key(e: dict[str, Any]) -> str:
            return str(e.get("date") or e.get("last_run") or "")
        return max(raw, key=_date_key)
    if isinstance(raw, dict):
        return raw
    return None


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Summarise Reflective Lantern run history")
    parser.add_argument(
        "--sort-by",
        choices=["commits", "date", "repo"],
        default="date",
        help="Sort rows by this column",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    rows: list[dict[str, Any]] = []
    for path in sorted(HISTORY_DIR.glob("*.json")):
        entry = load_latest_entry(path)
        if entry is None:
            continue
        rows.append({
            "repo": path.stem,
            "date": entry.get("date") or entry.get("last_run") or "",
            "mode": (entry.get("mode") or "improvement").upper(),
            "commits": entry.get("commits", 0),
            "tests_passed": entry.get("tests_passed", None),
        })

    if args.sort_by == "commits":
        rows.sort(key=lambda r: r["commits"], reverse=True)
    elif args.sort_by == "repo":
        rows.sort(key=lambda r: r["repo"])
    else:
        rows.sort(key=lambda r: r["date"], reverse=True)

    # Print table
    col_w = {"repo": 45, "date": 12, "mode": 12, "commits": 8, "tests_passed": 12}
    header = " | ".join(k.ljust(col_w[k]) for k in col_w)
    sep = "-+-".join("-" * col_w[k] for k in col_w)
    print(header)
    print(sep)
    for row in rows:
        tp = "yes" if row["tests_passed"] is True else ("no" if row["tests_passed"] is False else "-")
        print(
            f"{row['repo']:<{col_w['repo']}} | "
            f"{row['date']:<{col_w['date']}} | "
            f"{row['mode']:<{col_w['mode']}} | "
            f"{row['commits']:<{col_w['commits']}} | "
            f"{tp:<{col_w['tests_passed']}}"
        )
    total_commits = sum(r["commits"] for r in rows)
    print(f"\n{len(rows)} repos | {total_commits} total commits")


if __name__ == "__main__":
    main()
