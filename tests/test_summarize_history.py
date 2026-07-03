"""Tests for scripts.summarize_history."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def history_path(tmp_path: Path) -> Path:
    h = tmp_path / "history"
    h.mkdir()
    (h / "Alpha.json").write_text(json.dumps([
        {"date": "2026-06-30", "mode": "improvement", "commits": 60, "tests_passed": True},
        {"date": "2026-06-15", "mode": "improvement", "commits": 55, "tests_passed": False},
    ]))
    (h / "Beta.json").write_text(json.dumps(
        {"last_run": "2026-06-20", "commits": 30}
    ))
    return h


def test_load_latest_entry_picks_most_recent(history_path: Path) -> None:
    from scripts.summarize_history import load_latest_entry
    entry = load_latest_entry(history_path / "Alpha.json")
    assert entry is not None
    assert entry["date"] == "2026-06-30"
    assert entry["commits"] == 60


def test_load_latest_entry_handles_dict_format(history_path: Path) -> None:
    from scripts.summarize_history import load_latest_entry
    entry = load_latest_entry(history_path / "Beta.json")
    assert entry is not None
    assert entry["commits"] == 30


def test_load_latest_entry_returns_none_on_invalid(tmp_path: Path) -> None:
    from scripts.summarize_history import load_latest_entry
    f = tmp_path / "bad.json"
    f.write_text("{not valid")
    entry = load_latest_entry(f)
    assert entry is None


def test_load_latest_entry_returns_none_on_empty_list(tmp_path: Path) -> None:
    from scripts.summarize_history import load_latest_entry
    f = tmp_path / "empty.json"
    f.write_text("[]")
    entry = load_latest_entry(f)
    assert entry is None


@pytest.mark.parametrize("sort_by", ["commits", "date", "repo"])
def test_main_runs_without_error(history_path: Path, sort_by: str, capsys: pytest.CaptureFixture) -> None:
    import scripts.summarize_history as sh
    with patch.object(sh, "HISTORY_DIR", history_path):
        import sys
        with patch.object(sys, "argv", ["summarize_history.py", "--sort-by", sort_by]):
            sh.main()
    captured = capsys.readouterr()
    assert "Alpha" in captured.out
    assert "Beta" in captured.out


def test_load_latest_entry_returns_none_on_nested_invalid(tmp_path: Path) -> None:
    from scripts.summarize_history import load_latest_entry
    f = tmp_path / "nested.json"
    f.write_text("[[1, 2, 3]]")  # list of non-dicts
    entry = load_latest_entry(f)
    # Picks max by empty string key — returns the inner list, not a dict
    # This is a known edge case; function returns None for non-dict entries
    assert entry is None


@pytest.mark.parametrize("commits,expected_commits", [
    (60, 60),
    (0, 0),
    (120, 120),
])
def test_load_latest_entry_commit_values(tmp_path: Path, commits: int, expected_commits: int) -> None:
    from scripts.summarize_history import load_latest_entry
    f = tmp_path / "repo.json"
    f.write_text(json.dumps([{"date": "2026-07-01", "commits": commits}]))
    entry = load_latest_entry(f)
    assert entry is not None
    assert entry["commits"] == expected_commits


def test_main_skips_non_record_files(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """commit_schedule.json and schema.json should not appear in table output."""
    import scripts.summarize_history as sh
    h = tmp_path / "history"
    h.mkdir()
    (h / "MyRepo.json").write_text(json.dumps([{"date": "2026-07-01", "commits": 60}]))
    (h / "commit_schedule.json").write_text(json.dumps({
        "start_year": 2026, "start_week": 1
    }))
    import sys
    with (
        patch.object(sh, "HISTORY_DIR", h),
        patch.object(sys, "argv", ["sh.py"]),
    ):
        sh.main()
    out = capsys.readouterr().out
    assert "MyRepo" in out
    assert "commit_schedule" not in out
