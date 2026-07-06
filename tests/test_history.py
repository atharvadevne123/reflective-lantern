"""Tests for scripts.validate_history and history file schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_validate_entry_valid() -> None:
    from scripts.validate_history import validate_entry
    entry = {"date": "2026-06-01", "commits": 60, "mode": "improvement"}
    assert validate_entry(entry, "test.json", 0) == []


def test_validate_entry_missing_commits() -> None:
    from scripts.validate_history import validate_entry
    entry = {"date": "2026-06-01"}
    # commits missing — should pass if date present (commits optional in some formats)
    errors = validate_entry(entry, "test.json", 0)
    # date/last_run is present so no error there
    assert all("date" not in e for e in errors)


def test_validate_entry_no_date_fields() -> None:
    from scripts.validate_history import validate_entry
    entry = {"commits": 60, "mode": "improvement"}
    errors = validate_entry(entry, "test.json", 0)
    assert any("date" in e or "last_run" in e for e in errors)


def test_validate_entry_negative_commits() -> None:
    from scripts.validate_history import validate_entry
    entry = {"date": "2026-06-01", "commits": -5}
    errors = validate_entry(entry, "test.json", 0)
    assert any("negative" in e for e in errors)


def test_validate_entry_not_dict() -> None:
    from scripts.validate_history import validate_entry
    errors = validate_entry("not a dict", "test.json", 0)
    assert len(errors) == 1
    assert "dict" in errors[0]


def test_validate_file_valid_list(tmp_path: Path) -> None:
    from scripts.validate_history import validate_file
    data = [{"date": "2026-06-01", "commits": 60}]
    f = tmp_path / "repo.json"
    f.write_text(json.dumps(data))
    assert validate_file(f) == []


def test_validate_file_valid_dict(tmp_path: Path) -> None:
    from scripts.validate_history import validate_file
    data = {"last_run": "2026-06-01", "commits": 45}
    f = tmp_path / "repo.json"
    f.write_text(json.dumps(data))
    assert validate_file(f) == []


def test_validate_file_invalid_json(tmp_path: Path) -> None:
    from scripts.validate_history import validate_file
    f = tmp_path / "bad.json"
    f.write_text("{not valid")
    errors = validate_file(f)
    assert len(errors) == 1
    assert "JSON" in errors[0]


def test_validate_file_empty_list(tmp_path: Path) -> None:
    from scripts.validate_history import validate_file
    f = tmp_path / "empty.json"
    f.write_text("[]")
    assert validate_file(f) == []


@pytest.mark.parametrize("commits", [0, 1, 60, 300])
def test_validate_entry_valid_commit_counts(commits: int) -> None:
    from scripts.validate_history import validate_entry
    entry = {"date": "2026-06-01", "commits": commits}
    errors = validate_entry(entry, "test.json", 0)
    assert all("negative" not in e for e in errors)


def test_actual_history_files_valid() -> None:
    """All real history files in this repo must pass validation."""
    from scripts.validate_history import validate_file
    history_dir = Path(__file__).parent.parent / "history"
    _skip = {"schema.json", "commit_schedule.json"}
    errors: list[str] = []
    for path in sorted(p for p in history_dir.glob("*.json") if p.name not in _skip):
        errors.extend(validate_file(path))
    assert errors == [], "\n".join(errors)


def test_validate_file_multiple_entries(tmp_path: Path) -> None:
    from scripts.validate_history import validate_file
    data = [
        {"date": "2026-05-01", "commits": 60},
        {"date": "2026-06-01", "commits": 45},
        {"last_run": "2026-07-01", "commits": 30},
    ]
    f = tmp_path / "multi.json"
    f.write_text(json.dumps(data))
    assert validate_file(f) == []


def test_validate_entry_commits_wrong_type() -> None:
    from scripts.validate_history import validate_entry
    entry = {"date": "2026-06-01", "commits": "sixty"}
    errors = validate_entry(entry, "bad.json", 0)
    assert any("commits" in e for e in errors)


def test_validate_file_root_is_primitive(tmp_path: Path) -> None:
    from scripts.validate_history import validate_file
    f = tmp_path / "prim.json"
    f.write_text("42")
    errors = validate_file(f)
    assert len(errors) == 1
    assert "list or dict" in errors[0]


@pytest.mark.parametrize("mode", ["improvement", "IMPROVEMENT", "innovation", "INNOVATION"])
def test_validate_entry_mode_values(mode: str) -> None:
    from scripts.validate_history import validate_entry
    entry = {"date": "2026-06-01", "commits": 60, "mode": mode}
    errors = validate_entry(entry, "test.json", 0)
    assert errors == []


def test_validate_entry_valid_email_status() -> None:
    from scripts.validate_history import validate_entry
    for status in ("pending", "sent", "skipped", "failed_smtp", "network_blocked", "pdf_generated_ok"):
        entry = {"date": "2026-07-01", "commits": 60, "email_status": status}
        errors = validate_entry(entry, "test.json", 0)
        assert errors == [], f"Unexpected errors for email_status={status!r}: {errors}"


def test_validate_entry_invalid_email_status() -> None:
    from scripts.validate_history import validate_entry
    entry = {"date": "2026-07-01", "commits": 60, "email_status": "unknown_xyz"}
    errors = validate_entry(entry, "test.json", 0)
    assert any("email_status" in e for e in errors)


def test_validate_entry_no_email_status_ok() -> None:
    from scripts.validate_history import validate_entry
    entry = {"date": "2026-07-01", "commits": 60}
    errors = validate_entry(entry, "test.json", 0)
    assert errors == []
