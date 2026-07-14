"""Clean up old history entries from JSON history files."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from config.constants import CLEANUP_DEFAULT_DAYS, HISTORY_DIR, NON_RECORD_FILES


def _entry_date(entry: dict) -> date | None:
    """Return the date for *entry* using 'date' or 'last_run' field, or None."""
    raw = entry.get("date") or entry.get("last_run")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None


def clean_file(
    path: Path,
    cutoff: date,
    *,
    dry_run: bool = False,
    max_entries: int | None = None,
    min_keep: int = 0,
) -> int:
    """Remove entries older than *cutoff* from a history JSON file.

    Returns the number of entries removed.
    """
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0

    if isinstance(raw, dict):
        d = _entry_date(raw)
        if d is not None and d < cutoff:
            if not dry_run:
                path.write_text(json.dumps([]))
            return 1
        return 0

    if not isinstance(raw, list):
        return 0

    entries: list[Any] = [e for e in raw if isinstance(e, dict)]
    kept = []
    removed = 0
    for entry in entries:
        d = _entry_date(entry)
        if d is None or d >= cutoff:
            kept.append(entry)
        else:
            removed += 1

    if min_keep and len(kept) < min_keep:
        removed_entries = [e for e in entries if e not in kept]
        need = min_keep - len(kept)
        kept = kept + removed_entries[-need:]
        removed = len(entries) - len(kept)

    if max_entries is not None and len(kept) > max_entries:
        surplus = len(kept) - max_entries
        removed += surplus
        kept = kept[-max_entries:]

    if not dry_run:
        path.write_text(json.dumps(kept))
    return removed


def count_entries(directory: Path) -> dict[str, int]:
    """Return a mapping of filename → entry count for all JSON files in *directory*."""
    result: dict[str, int] = {}
    for f in directory.glob("*.json"):
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            result[f.name] = 0
            continue
        if isinstance(raw, list):
            result[f.name] = len(raw)
        elif isinstance(raw, dict):
            result[f.name] = 1
        else:
            result[f.name] = 0
    return result


def main() -> int:
    import argparse
    from datetime import timedelta

    parser = argparse.ArgumentParser(description="Clean up old history entries.")
    parser.add_argument("--days", type=int, default=CLEANUP_DEFAULT_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("directory", nargs="?", default=None)
    args = parser.parse_args()

    directory = Path(args.directory) if args.directory else HISTORY_DIR
    cutoff = date.today() - timedelta(days=args.days)
    total = 0
    for f in directory.glob("*.json"):
        if f.stem in NON_RECORD_FILES:
            continue
        removed = clean_file(f, cutoff, dry_run=args.dry_run)
        if removed:
            print(f"  {f.name}: removed {removed} entries")
        total += removed
    print(f"Total removed: {total}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
