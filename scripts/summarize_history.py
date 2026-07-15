"""Summarize Reflective Lantern run history across all repos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from config.constants import HISTORY_DIR, NON_RECORD_FILES

_NON_RECORD_STEMS = {s.replace(".json", "") for s in NON_RECORD_FILES}


def load_all_entries(path: Path) -> list[dict]:
    """Return all dict entries from a history JSON file."""
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    return []


def load_latest_entry(path: Path) -> dict | None:
    """Return the most-recent entry from a history file, or None."""
    entries = load_all_entries(path)
    if not entries:
        return None

    def _sort_key(e: dict) -> str:
        return str(e.get("date") or e.get("last_run") or "")

    best = max(entries, key=_sort_key)
    if not isinstance(best, dict):
        return None
    return best


def _collect_rows(history_dir: Path, filter_mode: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(history_dir.glob("*.json")):
        if f.stem in _NON_RECORD_STEMS:
            continue
        entry = load_latest_entry(f)
        if entry is None:
            continue
        mode = str(entry.get("mode", "")).lower()
        if filter_mode and mode != filter_mode.lower():
            continue
        rows.append(
            {
                "repo": f.stem,
                "date": entry.get("date") or entry.get("last_run") or "",
                "commits": entry.get("commits", 0),
                "mode": mode,
                "tests_passed": entry.get("tests_passed"),
            }
        )
    return rows


def main() -> int | None:
    parser = argparse.ArgumentParser(description="Summarize run history.")
    parser.add_argument("--sort-by", choices=["commits", "date", "repo"], default="date")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--filter-mode", default=None)
    args = parser.parse_args()

    rows = _collect_rows(HISTORY_DIR, filter_mode=args.filter_mode)

    if args.sort_by == "commits":
        rows.sort(key=lambda r: r["commits"], reverse=True)
    elif args.sort_by == "date":
        rows.sort(key=lambda r: r["date"], reverse=True)
    else:
        rows.sort(key=lambda r: r["repo"])

    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("No history found.")
        return 0

    print(f"{'Repo':<30} {'Date':<12} {'Commits':>8} {'Mode':<15} {'Tests':>6}")
    print("-" * 75)
    for r in rows:
        tp = "✓" if r["tests_passed"] else ("✗" if r["tests_passed"] is False else "?")
        print(f"{r['repo']:<30} {r['date']:<12} {r['commits']:>8} {r['mode']:<15} {tp:>6}")
    print(f"\n{len(rows)} repos shown.")
    return 0


def export_to_csv(history_dir: Path | None = None, output_path: Path | None = None) -> int:
    """Write history summary to a CSV file.

    Args:
        history_dir: Directory containing per-repo JSON files (default: HISTORY_DIR).
        output_path: Destination CSV file (default: history_dir / "summary.csv").

    Returns:
        Number of rows written.
    """
    import csv

    history_dir = history_dir or HISTORY_DIR
    rows = _collect_rows(history_dir)
    if not rows:
        return 0

    dest = output_path or (history_dir / "summary.csv")
    fieldnames = ["repo", "date", "commits", "mode", "tests_passed"]
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


__all__ = ["export_to_csv", "load_all_entries", "load_latest_entry"]


if __name__ == "__main__":
    sys.exit(main() or 0)
