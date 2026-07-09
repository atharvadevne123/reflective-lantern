#!/usr/bin/env python3
"""Remove history entries older than N days and truncate bloated files.

Usage:
    python scripts/cleanup.py --days 90 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from config.constants import CLEANUP_DEFAULT_DAYS

log = logging.getLogger(__name__)
HISTORY_DIR = Path(__file__).parent.parent / "history"


def _entry_date(entry: dict[str, Any]) -> date | None:
    """Extract and parse the date from a history entry."""
    raw = entry.get("date") or entry.get("last_run")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def count_entries(history_dir: Path) -> dict[str, int]:
    """Return a mapping of filename → entry count for all JSON files in *history_dir*."""
    result: dict[str, int] = {}
    for path in sorted(history_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, list):
                result[path.name] = len(raw)
            elif isinstance(raw, dict):
                result[path.name] = 1
        except Exception:
            result[path.name] = 0
    return result


def clean_file(
    path: Path,
    cutoff: date,
    dry_run: bool = False,
    max_entries: int | None = None,
    min_keep: int = 0,
) -> int:
    """Remove entries older than *cutoff* from *path*. Return number removed.

    If *max_entries* is set, also truncate the file to the most recent
    *max_entries* entries after the date-based removal.
    """
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        log.warning("Skipping %s: %s", path.name, exc)
        return 0

    if isinstance(raw, dict):
        entry_date = _entry_date(raw)
        if entry_date and entry_date < cutoff:
            log.info("Would remove single-entry %s (dated %s)", path.name, entry_date)
            return 1
        return 0

    if not isinstance(raw, list):
        return 0

    original_count = len(raw)
    kept = [e for e in raw if not ((d := _entry_date(e)) is not None and d < cutoff)]
    # Always preserve at least min_keep most-recent entries if requested
    if min_keep > 0 and len(kept) < min_keep and len(raw) >= min_keep:
        kept = raw[-min_keep:]

    if max_entries is not None and len(kept) > max_entries:
        kept = kept[-max_entries:]

    removed = original_count - len(kept)

    if removed > 0:
        log.info("%s: removing %d old entry/entries", path.name, removed)
        if not dry_run:
            path.write_text(json.dumps(kept, indent=2))

    return removed


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Clean old history entries")
    parser.add_argument(
        "--days",
        type=int,
        default=CLEANUP_DEFAULT_DAYS,
        help="Remove entries older than this many days",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying files")
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Keep at most this many entries per file after date pruning",
    )
    parser.add_argument(
        "--min-keep",
        type=int,
        default=0,
        help="Always keep at least this many of the most recent entries (default: 0 = no minimum)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cutoff = date.today() - timedelta(days=args.days)
    log.info("Cutoff date: %s", cutoff.isoformat())

    total_removed = 0
    for path in sorted(HISTORY_DIR.glob("*.json")):
        total_removed += clean_file(
            path,
            cutoff,
            dry_run=args.dry_run,
            max_entries=args.max_entries,
            min_keep=args.min_keep,
        )

    action = "Would remove" if args.dry_run else "Removed"
    log.info("%s %d old entries from history files", action, total_removed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
