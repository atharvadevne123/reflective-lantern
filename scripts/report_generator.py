#!/usr/bin/env python3
"""Generate a Markdown daily/weekly summary from history JSON files.

Usage:
    python scripts/report_generator.py --mode daily  --date 2026-07-01
    python scripts/report_generator.py --mode weekly
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from config.constants import NON_RECORD_FILES

log = logging.getLogger(__name__)
HISTORY_DIR = Path(__file__).parent.parent / "history"


def format_date(d: date) -> str:
    """Return *d* formatted as YYYY-MM-DD."""
    return d.strftime("%Y-%m-%d")


def total_commits(history: dict[str, list[dict[str, Any]]]) -> int:
    """Return the sum of all commit counts across all repos and entries."""
    return sum(
        int(entry.get("commits", 0))
        for entries in history.values()
        for entry in entries
        if isinstance(entry, dict)
    )


def load_all_history() -> dict[str, list[dict[str, Any]]]:
    """Load every run-record JSON file from the history directory."""
    history: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(p for p in HISTORY_DIR.glob("*.json") if p.name not in NON_RECORD_FILES):
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                history[path.stem] = data
            elif isinstance(data, dict):
                history[path.stem] = [data]
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)
    return history


def daily_report(target_date: date) -> str:
    """Return a Markdown string summarising all runs on *target_date*."""
    ds = target_date.isoformat()
    history = load_all_history()
    lines: list[str] = [
        f"# Reflective Lantern — Daily Report ({ds})",
        "",
    ]
    found = False
    for repo, entries in history.items():
        for entry in entries:
            if entry.get("date") == ds or entry.get("last_run") == ds:
                found = True
                commits = entry.get("commits", "?")
                mode = entry.get("mode", "improvement").upper()
                lines += [
                    f"## {repo} [{mode}]",
                    f"- **Commits**: {commits}",
                ]
                if "tiers" in entry:
                    for tier, items in entry["tiers"].items():
                        lines.append(f"- **{tier}**:")
                        for item in items:
                            lines.append(f"  - {item}")
                if "improvements" in entry:
                    lines.append("- **Changes**:")
                    for imp in entry["improvements"]:
                        lines.append(f"  - {imp}")
                lines.append("")
    if not found:
        lines.append(f"_No runs found for {ds}._")
    return "\n".join(lines)


def weekly_report(end_date: date | None = None, window_days: int = 7) -> str:
    """Return a Markdown string summarising the past *window_days* days."""
    end = end_date or date.today()
    start = end - timedelta(days=window_days - 1)
    history = load_all_history()
    total_commits = 0
    repos_touched: set[str] = set()
    lines: list[str] = [
        f"# Reflective Lantern — Weekly Summary ({start.isoformat()} – {end.isoformat()})",
        "",
        "| Repository | Date | Mode | Commits |",
        "| --- | --- | --- | --- |",
    ]
    for repo, entries in sorted(history.items()):
        for entry in entries:
            entry_date_str = entry.get("date") or entry.get("last_run") or ""
            try:
                entry_date = date.fromisoformat(entry_date_str)
            except ValueError:
                continue
            if start <= entry_date <= end:
                commits = entry.get("commits", 0)
                mode = entry.get("mode", "improvement").upper()
                lines.append(f"| {repo} | {entry_date_str} | {mode} | {commits} |")
                total_commits += int(commits)
                repos_touched.add(repo)

    lines += [
        "",
        f"**Total commits this week**: {total_commits}",
        f"**Repos touched**: {len(repos_touched)}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate Reflective Lantern Markdown reports")
    parser.add_argument(
        "--mode", choices=["daily", "weekly"], default="daily", help="Report type"
    )
    parser.add_argument("--date", help="Target date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--output", help="Write report to this file (default: stdout)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    target = date.fromisoformat(args.date) if args.date else date.today()
    report = daily_report(target) if args.mode == "daily" else weekly_report(target)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
