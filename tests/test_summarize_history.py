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
