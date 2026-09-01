"""Parametrized tests that validate every real history file in history/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.constants import NON_RECORD_FILES

HISTORY_DIR = Path(__file__).parent.parent / "history"
_EXCLUDED = NON_RECORD_FILES
HISTORY_FILES = sorted(p for p in HISTORY_DIR.glob("*.json") if p.name not in _EXCLUDED)


@pytest.mark.parametrize("history_file", HISTORY_FILES, ids=lambda p: p.stem)
def test_history_file_is_valid_json(history_file: Path) -> None:
    """Every history file must be parseable JSON."""
    content = history_file.read_text(encoding="utf-8")
    parsed = json.loads(content)  # raises on invalid JSON
    assert parsed is not None


@pytest.mark.parametrize("history_file", HISTORY_FILES, ids=lambda p: p.stem)
def test_history_file_has_commits_field(history_file: Path) -> None:
    """Every entry must have a non-negative commits field."""
    data = json.loads(history_file.read_text())
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        assert "commits" in entry, f"{history_file.name}: missing 'commits'"
        assert isinstance(entry["commits"], int), f"{history_file.name}: commits not int"
        assert entry["commits"] >= 0, f"{history_file.name}: commits is negative"


@pytest.mark.parametrize("history_file", HISTORY_FILES, ids=lambda p: p.stem)
def test_history_file_has_date_field(history_file: Path) -> None:
    """Every entry must have a date or last_run field."""
    data = json.loads(history_file.read_text())
    entries = data if isinstance(data, list) else [data]
    for i, entry in enumerate(entries):
        has_date = "date" in entry or "last_run" in entry
        assert has_date, f"{history_file.name}[{i}]: missing 'date' or 'last_run'"


@pytest.mark.parametrize("history_file", HISTORY_FILES, ids=lambda p: p.stem)
def test_history_file_commits_below_reasonable_cap(history_file: Path) -> None:
    """Commits should be below 1000 (sanity check for data corruption)."""
    data = json.loads(history_file.read_text())
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        commits = entry.get("commits", 0)
        assert commits < 1000, f"{history_file.name}: suspiciously high commits={commits}"


@pytest.mark.parametrize("history_file", HISTORY_FILES, ids=lambda p: p.stem)
def test_history_file_mode_if_present_is_valid(history_file: Path) -> None:
    """If a mode field is present, it must be 'improvement' or 'innovation'."""
    data = json.loads(history_file.read_text())
    entries = data if isinstance(data, list) else [data]
    valid_modes = {"improvement", "innovation", "IMPROVEMENT", "INNOVATION", "user-requested"}
    for i, entry in enumerate(entries):
        if "mode" in entry:
            assert entry["mode"] in valid_modes, f"{history_file.name}[{i}]: unexpected mode={entry['mode']!r}"


@pytest.mark.parametrize("history_file", HISTORY_FILES, ids=lambda p: p.stem)
def test_history_file_dates_are_iso_format(history_file: Path) -> None:
    """All date and last_run values must parse as ISO dates."""
    from datetime import date

    data = json.loads(history_file.read_text())
    entries = data if isinstance(data, list) else [data]
    for i, entry in enumerate(entries):
        for field in ("date", "last_run"):
            if field in entry:
                val = entry[field]
                try:
                    date.fromisoformat(str(val))
                except ValueError:
                    pytest.fail(f"{history_file.name}[{i}]: {field}={val!r} is not ISO format")


@pytest.mark.parametrize("history_file", HISTORY_FILES, ids=lambda p: p.stem)
def test_history_file_email_status_if_present_is_valid(history_file: Path) -> None:
    """email_status must be a recognised value when present."""
    from scripts.validate_history import VALID_EMAIL_STATUS_PREFIXES, VALID_EMAIL_STATUSES

    data = json.loads(history_file.read_text())
    entries = data if isinstance(data, list) else [data]
    for i, entry in enumerate(entries):
        if "email_status" in entry and isinstance(entry["email_status"], str):
            status = entry["email_status"]
            valid = status in VALID_EMAIL_STATUSES or any(status.startswith(p) for p in VALID_EMAIL_STATUS_PREFIXES)
            assert valid, f"{history_file.name}[{i}]: unrecognised email_status={status!r}"


@pytest.mark.parametrize("history_file", HISTORY_FILES, ids=lambda p: p.stem)
def test_history_file_commit_count_non_negative(history_file: Path) -> None:
    """commit_count must be a non-negative integer when present."""
    data = json.loads(history_file.read_text())
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        if "commit_count" in entry:
            assert isinstance(entry["commit_count"], int)
            assert entry["commit_count"] >= 0
