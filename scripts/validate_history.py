"""Validate history JSON files against the expected schema."""

from __future__ import annotations

import json
from pathlib import Path

from config.constants import HISTORY_DIR, NON_RECORD_FILES

VALID_MODES: frozenset[str] = frozenset({"improvement", "IMPROVEMENT", "innovation", "INNOVATION", "user-requested"})
VALID_EMAIL_STATUSES: frozenset[str] = frozenset(
    {"pending", "sent", "skipped", "failed_smtp", "network_blocked", "pdf_generated_ok", ""}
)
VALID_EMAIL_STATUS_PREFIXES: tuple[str, ...] = (
    "pending",
    "sent",
    "skipped",
    "failed_smtp",
    "failed_",
    "network_blocked",
    "pdf_generated_ok",
    "smtp_",
)


def validate_entry(entry: object, filename: str, index: int) -> list[str]:
    """Validate a single history entry dict. Returns list of error strings."""
    if not isinstance(entry, dict):
        return [f"{filename}[{index}]: expected dict, got {type(entry).__name__}"]

    errors: list[str] = []

    if "date" not in entry and "last_run" not in entry:
        errors.append(f"{filename}[{index}]: missing 'date' or 'last_run' field")

    commits = entry.get("commits")
    if commits is not None:
        if not isinstance(commits, int):
            errors.append(f"{filename}[{index}]: commits must be an int, got {type(commits).__name__}")
        elif commits < 0:
            errors.append(f"{filename}[{index}]: negative commits value ({commits})")

    mode = entry.get("mode")
    if mode is not None and str(mode) not in VALID_MODES:
        errors.append(f"{filename}[{index}]: invalid mode {mode!r}")

    email_status = entry.get("email_status")
    if email_status is not None and isinstance(email_status, str):
        is_valid = email_status in VALID_EMAIL_STATUSES or any(
            email_status.startswith(p) for p in VALID_EMAIL_STATUS_PREFIXES
        )
        if not is_valid:
            errors.append(f"{filename}[{index}]: invalid email_status {email_status!r}")

    return errors


def validate_file(path: Path) -> list[str]:
    """Validate all entries in a history JSON file. Returns list of error strings."""
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{path.name}: JSON parse error: {exc}"]

    if isinstance(raw, list):
        errors: list[str] = []
        for i, entry in enumerate(raw):
            errors.extend(validate_entry(entry, path.name, i))
        return errors
    elif isinstance(raw, dict):
        return validate_entry(raw, path.name, 0)
    else:
        return [f"{path.name}: root must be list or dict, got {type(raw).__name__}"]


def validate_files(paths: list[Path]) -> dict[str, list[str]]:
    """Validate multiple history files. Returns mapping filename → errors."""
    return {p.name: validate_file(p) for p in paths}


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate history JSON files.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("directory", nargs="?", default=None)
    args = parser.parse_args()

    directory = Path(args.directory) if args.directory else HISTORY_DIR
    files = sorted(f for f in directory.glob("*.json") if f.name not in NON_RECORD_FILES)

    if args.as_json:
        results = []
        for f in files:
            errs = validate_file(f)
            results.append({"file": f.name, "errors": errs})
        print(json.dumps(results, indent=2))
        has_errors = any(r["errors"] for r in results)
        return 1 if has_errors else 0

    all_errors: list[str] = []
    for f in files:
        errs = validate_file(f)
        if errs:
            print(f"  FAIL {f.name}:")
            for e in errs:
                print(f"    {e}")
            all_errors.extend(errs)
        else:
            if args.verbose:
                print(f"  OK   {f.name}")
    if all_errors:
        print(f"\n{len(all_errors)} error(s) found.", file=sys.stderr)
        return 1
    if not args.verbose:
        print(f"All {len(files)} files valid.")
    return 0


__all__ = [
    "VALID_EMAIL_STATUSES",
    "VALID_EMAIL_STATUS_PREFIXES",
    "VALID_MODES",
    "validate_entry",
    "validate_file",
    "validate_files",
]


if __name__ == "__main__":
    import sys

    sys.exit(main())
