"""Tests for scripts.cleanup."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest


@pytest.fixture()
def history_path(tmp_path: Path) -> Path:
    return tmp_path / "history"


def write_history(path: Path, entries: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))
    return path


def test_clean_removes_old_entries(tmp_path: Path) -> None:
    from scripts.cleanup import clean_file
    f = write_history(tmp_path / "repo.json", [
        {"date": "2020-01-01", "commits": 5},
        {"date": "2026-06-30", "commits": 60},
    ])
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 1
    kept = json.loads(f.read_text())
    assert len(kept) == 1
    assert kept[0]["date"] == "2026-06-30"


def test_clean_dry_run_does_not_modify(tmp_path: Path) -> None:
    from scripts.cleanup import clean_file
    original = [{"date": "2020-01-01", "commits": 5}]
    f = write_history(tmp_path / "repo.json", original)
    removed = clean_file(f, cutoff=date(2026, 1, 1), dry_run=True)
    assert removed == 1
    assert json.loads(f.read_text()) == original


def test_clean_keeps_recent_entries(tmp_path: Path) -> None:
    from scripts.cleanup import clean_file
    f = write_history(tmp_path / "repo.json", [
        {"date": "2026-06-01", "commits": 60},
        {"date": "2026-06-15", "commits": 60},
    ])
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 0


def test_clean_handles_invalid_json(tmp_path: Path) -> None:
    from scripts.cleanup import clean_file
    f = tmp_path / "bad.json"
    f.write_text("{not valid")
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 0


def test_clean_handles_missing_date_field(tmp_path: Path) -> None:
    from scripts.cleanup import clean_file
    f = write_history(tmp_path / "repo.json", [
        {"commits": 60},  # no date field — should not be removed
    ])
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 0


def test_clean_single_dict_format(tmp_path: Path) -> None:
    from scripts.cleanup import clean_file
    f = tmp_path / "old.json"
    f.write_text(json.dumps({"last_run": "2020-01-01", "commits": 5}))
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 1


@pytest.mark.parametrize("days,expected_removed", [(30, 1), (3650, 1)])
def test_clean_parametrized(tmp_path: Path, days: int, expected_removed: int) -> None:
    from datetime import timedelta

    from scripts.cleanup import clean_file
    old_date = (date.today() - timedelta(days=days + 1)).isoformat()
    f = write_history(tmp_path / "repo.json", [{"date": old_date, "commits": 5}])
    cutoff = date.today() - timedelta(days=days)
    removed = clean_file(f, cutoff=cutoff)
    assert removed == expected_removed

def test_clean_file_last_run_field(tmp_path: Path) -> None:
    """Entries using 'last_run' instead of 'date' should also be cleaned."""
    from scripts.cleanup import clean_file
    f = write_history(tmp_path / "repo.json", [
        {"last_run": "2020-06-01", "commits": 10},
        {"last_run": "2026-06-30", "commits": 60},
    ])
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 1
    kept = json.loads(f.read_text())
    assert len(kept) == 1


def test_clean_file_mixed_date_formats(tmp_path: Path) -> None:
    """Entries with both date and last_run should use 'date' field."""
    from scripts.cleanup import clean_file
    f = write_history(tmp_path / "repo.json", [
        {"date": "2025-01-01", "last_run": "2026-06-30", "commits": 5},
    ])
    # 'date' field is 2025-01-01, which is older than cutoff 2026-01-01
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 1


@pytest.mark.parametrize("content", ["null", '"string"', "42"])
def test_clean_file_non_list_non_dict(tmp_path: Path, content: str) -> None:
    """Non-list, non-dict JSON root should return 0 removals."""
    from scripts.cleanup import clean_file
    f = tmp_path / "repo.json"
    f.write_text(content)
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 0


def test_clean_file_empty_list(tmp_path: Path) -> None:
    from scripts.cleanup import clean_file
    f = write_history(tmp_path / "repo.json", [])
    removed = clean_file(f, cutoff=date(2026, 1, 1))
    assert removed == 0
