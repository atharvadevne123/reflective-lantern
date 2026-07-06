#!/usr/bin/env python3
"""Validate all history JSON files against the expected schema.

Usage:
    python scripts/validate_history.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
HISTORY_DIR = Path(__file__).parent.parent / "history"
from config.constants import NON_RECORD_FILES  # noqa: E402

REQUIRED_FIELDS: dict[str, type] = {
    "date": str,
    "commits": int,
}

OPTIONAL_FIELDS: dict[str, type] = {
    "mode": str,
    "tests_passed": bool,
    "improvements": list,
    "tiers": dict,
    "email_status": str,
    "last_run": str,
    "repo": str,
}

VALID_EMAIL_STATUSES: frozenset[str] = frozenset({"pending", "sent", "skipped"})
VALID_EMAIL_STATUS_PREFIXES: tuple[str, ...] = ("failed", "network_", "pdf_generated")


def validate_entry(entry: Any, source: str, idx: int) -> list[str]:
    """Return a list of validation error strings for a single history entry."""
    errors: list[str] = []
    if not isinstance(entry, dict):
        return [f"{source}[{idx}]: expected dict, got {type(entry).__name__}"]

    for field_name, expected_type in REQUIRED_FIELDS.items():
        if field_name not in entry and field_name not in (
            "date",  # some entries use last_run
        ):
            errors.append(f"{source}[{idx}]: missing required field '{field_name}'")
        elif field_name in entry and not isinstance(entry[field_name], expected_type):
            errors.append(
                f"{source}[{idx}]: '{field_name}' should be {expected_type.__name__}, "
                f"got {type(entry[field_name]).__name__}"
            )

    # Ensure at least one date field
    if "date" not in entry and "last_run" not in entry:
        errors.append(f"{source}[{idx}]: must have 'date' or 'last_run'")

    # Commits should be non-negative
    if "commits" in entry and isinstance(entry["commits"], int):
        if entry["commits"] < 0:
            errors.append(f"{source}[{idx}]: 'commits' is negative")

    # Validate email_status if present
    if "email_status" in entry and isinstance(entry["email_status"], str):
        status = entry["email_status"]
        if status not in VALID_EMAIL_STATUSES and not any(
            status.startswith(p) for p in VALID_EMAIL_STATUS_PREFIXES
        ):
            errors.append(
                f"{source}[{idx}]: unrecognised email_status '{status}'"
            )

    return errors


def validate_file(path: Path) -> list[str]:
    """Validate one history JSON file. Return list of error strings."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path.name}: invalid JSON — {exc}"]

    if isinstance(raw, dict):
        # Single-entry format
        return validate_entry(raw, path.name, 0)
    if isinstance(raw, list):
        errors: list[str] = []
        for i, entry in enumerate(raw):
            errors.extend(validate_entry(entry, path.name, i))
        return errors
    return [f"{path.name}: expected list or dict at root"]


def main() -> int:
    """Validate all history files. Return 0 if all valid, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Validate Reflective Lantern history files")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print each file as it is checked"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as a JSON array"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    all_errors: list[str] = []
    file_results: list[dict[str, object]] = []
    files = sorted(p for p in HISTORY_DIR.glob("*.json") if p.name not in NON_RECORD_FILES)
    for path in files:
        if args.verbose and not args.json:
            print(f"Checking {path.name} …", end=" ", flush=True)
        errors = validate_file(path)
        if args.verbose and not args.json:
            print("OK" if not errors else f"FAIL ({len(errors)} error(s))")
        all_errors.extend(errors)
        file_results.append({"file": path.name, "errors": errors})

    if args.json:
        import json as _json
        print(_json.dumps(file_results, indent=2))
        return 1 if all_errors else 0

    if all_errors:
        for err in all_errors:
            log.error(err)
        print(f"\n{len(files)} files checked, {len(all_errors)} error(s) found")
        return 1

    print(f"{len(files)} history files validated — all OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
