"""Generate daily and weekly reports from run history."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from config.constants import HISTORY_DIR, NON_RECORD_FILES, SEPARATOR_WIDTH

_NON_RECORD_STEMS = {s.replace(".json", "") for s in NON_RECORD_FILES}


def format_date(d: date) -> str:
    """Return ISO format date string."""
    return d.isoformat()


def load_all_history(history_dir: Path | None = None) -> dict[str, list[dict]]:
    """Load all repo history files into a dict mapping repo name → list of entries."""
    directory = history_dir or HISTORY_DIR
    result: dict[str, list[dict]] = {}
    for f in directory.glob("*.json"):
        if f.stem in _NON_RECORD_STEMS:
            continue
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(raw, dict):
            result[f.stem] = [raw]
        elif isinstance(raw, list):
            result[f.stem] = [e for e in raw if isinstance(e, dict)]
        else:
            result[f.stem] = []
    return result


def total_commits(history: dict[str, list[dict]]) -> int:
    """Return the total number of commits across all repos and entries."""
    return sum(e.get("commits", 0) for entries in history.values() for e in entries)


def _entry_date_str(entry: dict) -> str:
    return str(entry.get("date") or entry.get("last_run") or "")


def daily_report(target: date) -> str:
    """Generate a daily report for *target* date."""
    history = load_all_history()
    target_str = format_date(target)
    lines = [f"Daily Report — {target_str}", "=" * SEPARATOR_WIDTH]
    found: list[tuple[str, dict[str, Any]]] = []
    for repo, entries in history.items():
        for e in entries:
            if _entry_date_str(e) == target_str:
                found.append((repo, e))
    if not found:
        lines.append("No runs found for this date.")
        return "\n".join(lines)
    for repo, e in found:
        mode = str(e.get("mode", "")).upper()
        commits = e.get("commits", 0)
        tests = "PASS" if e.get("tests_passed") else "FAIL"
        lines.append(f"\n  {repo}")
        lines.append(f"    Mode:    {mode}")
        lines.append(f"    Commits: {commits}")
        lines.append(f"    Tests:   {tests}")
        improvements = e.get("improvements")
        if improvements:
            for imp in improvements:
                if isinstance(imp, str):
                    lines.append(f"    - {imp}")
    return "\n".join(lines)


def weekly_report(target: date, window_days: int = 7) -> str:
    """Generate a weekly report for the *window_days* period ending at *target*."""
    history = load_all_history()
    start = target - timedelta(days=window_days - 1)
    start_str = format_date(start)
    target_str = format_date(target)

    lines = [
        f"Weekly Summary — {start_str} to {target_str}",
        "=" * SEPARATOR_WIDTH,
        f"{'Repository':<30} {'Commits':>8} {'Runs':>5}",
        "-" * SEPARATOR_WIDTH,
    ]
    summary_commits = 0
    repos_touched = 0
    for repo, entries in sorted(history.items()):
        window_entries = [e for e in entries if start_str <= _entry_date_str(e) <= target_str]
        if not window_entries:
            continue
        c = sum(e.get("commits", 0) for e in window_entries)
        lines.append(f"{repo:<30} {c:>8} {len(window_entries):>5}")
        summary_commits += c
        repos_touched += 1

    lines.append("-" * SEPARATOR_WIDTH)
    lines.append(f"Total commits: {summary_commits}")
    lines.append(f"Repos touched: {repos_touched}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate run history reports.")
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--date", default=format_date(date.today()))
    parser.add_argument("--output", default=None, help="Write report to file instead of stdout")
    args = parser.parse_args()

    target = date.fromisoformat(args.date)
    if args.mode == "weekly":
        content = weekly_report(target)
    else:
        content = daily_report(target)

    if args.output:
        Path(args.output).write_text(content)
    else:
        print(content)
    return 0


__all__ = ["daily_report", "format_date", "load_all_history", "total_commits", "weekly_report"]


if __name__ == "__main__":
    sys.exit(main())
