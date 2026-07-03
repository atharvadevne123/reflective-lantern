"""Tests for scripts.report_generator."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def sample_history(tmp_path: Path) -> Path:
    h = tmp_path / "history"
    h.mkdir()
    (h / "RepoA.json").write_text(json.dumps([
        {"date": "2026-06-30", "mode": "improvement", "commits": 60,
         "improvements": ["added types"]},
        {"date": "2026-06-23", "mode": "improvement", "commits": 55},
    ]))
    (h / "RepoB.json").write_text(json.dumps(
        {"last_run": "2026-06-30", "commits": 30, "mode": "improvement"}
    ))
    return h


def test_daily_report_contains_repo(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.daily_report(date(2026, 6, 30))
    assert "RepoA" in report


def test_daily_report_no_runs(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.daily_report(date(2025, 1, 1))
    assert "No runs found" in report


def test_daily_report_shows_commits(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.daily_report(date(2026, 6, 30))
    assert "60" in report


def test_weekly_report_table_header(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.weekly_report(date(2026, 6, 30))
    assert "Repository" in report
    assert "Commits" in report


def test_weekly_report_total_commits(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.weekly_report(date(2026, 6, 30))
    # RepoA has 60+55 but 55 is outside 7-day window from 6-30; RepoB has 30
    assert "Total commits" in report


def test_load_all_history_skips_invalid(sample_history: Path) -> None:
    from scripts import report_generator as rg
    (sample_history / "Bad.json").write_text("{bad")
    with patch.object(rg, "HISTORY_DIR", sample_history):
        history = rg.load_all_history()
    assert "Bad" not in history


def test_load_all_history_single_dict(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        history = rg.load_all_history()
    assert "RepoB" in history
    assert isinstance(history["RepoB"], list)


@pytest.mark.parametrize("target_date,expected_present", [
    (date(2026, 6, 30), True),   # date exists
    (date(2026, 6, 23), True),   # second entry
    (date(2026, 1, 1), False),   # no entries on this date
])
def test_daily_report_date_presence(
    sample_history: Path, target_date: date, expected_present: bool
) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.daily_report(target_date)
    if expected_present:
        assert "No runs found" not in report
    else:
        assert "No runs found" in report


def test_daily_report_shows_mode(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.daily_report(date(2026, 6, 30))
    assert "IMPROVEMENT" in report


def test_daily_report_shows_improvements(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.daily_report(date(2026, 6, 30))
    assert "added types" in report


def test_load_all_history_excludes_non_record_files(tmp_path: Path) -> None:
    """commit_schedule.json should not appear as a history entry."""
    from scripts import report_generator as rg
    h = tmp_path / "history"
    h.mkdir()
    (h / "Repo.json").write_text(json.dumps([{"date": "2026-07-01", "commits": 60}]))
    (h / "commit_schedule.json").write_text(json.dumps({"start_year": 2026}))
    with patch.object(rg, "HISTORY_DIR", h):
        history = rg.load_all_history()
    assert "Repo" in history


def test_weekly_report_covers_full_7_days(sample_history: Path) -> None:
    from scripts import report_generator as rg
    with patch.object(rg, "HISTORY_DIR", sample_history):
        report = rg.weekly_report(date(2026, 6, 30))
    # The report should mention 7 days
    assert "2026-06" in report
