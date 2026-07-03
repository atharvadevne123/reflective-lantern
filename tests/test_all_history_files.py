"""Parametrized tests that validate every real history file in history/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

HISTORY_DIR = Path(__file__).parent.parent / "history"
# commit_schedule.json is a config file; schema.json is a JSON schema — neither are run records
_EXCLUDED = {"schema.json", "commit_schedule.json"}
HISTORY_FILES = sorted(
    p for p in HISTORY_DIR.glob("*.json") if p.name not in _EXCLUDED
)


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
