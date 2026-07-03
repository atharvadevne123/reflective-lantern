"""Integration tests: validate full script entry-points without network calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_validate_history_main_passes_on_good_files(tmp_path: Path) -> None:
    """validate_history.main() exits 0 when all files are valid."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "GoodRepo.json").write_text(json.dumps([{"date": "2026-06-01", "commits": 60}]))
    import scripts.validate_history as vh
    with patch.object(vh, "HISTORY_DIR", h):
        result = vh.main()
    assert result == 0


def test_validate_history_main_fails_on_bad_files(tmp_path: Path) -> None:
    """validate_history.main() exits 1 when a file has no date fields."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "BadRepo.json").write_text(json.dumps([{"commits": 60}]))
    import scripts.validate_history as vh
    with patch.object(vh, "HISTORY_DIR", h):
        result = vh.main()
    assert result == 1


def test_cleanup_main_dry_run_exits_zero(tmp_path: Path) -> None:
    """cleanup.main() exits 0 in dry-run mode."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "OldRepo.json").write_text(json.dumps([{"date": "2020-01-01", "commits": 10}]))
    import scripts.cleanup as cl
    with patch.object(cl, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["cleanup.py", "--dry-run", "--days", "30"]):
            result = cl.main()
    assert result == 0


def test_report_generator_daily_returns_string(tmp_path: Path) -> None:
    """daily_report() returns a non-empty Markdown string."""
    from datetime import date
    h = tmp_path / "history"
    h.mkdir()
    (h / "MyRepo.json").write_text(json.dumps([
        {"date": "2026-07-01", "commits": 60, "improvements": ["added tests"]}
    ]))
    import scripts.report_generator as rg
    with patch.object(rg, "HISTORY_DIR", h):
        report = rg.daily_report(date(2026, 7, 1))
    assert isinstance(report, str)
    assert len(report) > 0
    assert "MyRepo" in report


def test_summarize_history_main_prints_table(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """summarize_history.main() prints a table with repo names."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "Alpha.json").write_text(json.dumps([{"date": "2026-06-30", "commits": 60}]))
    import scripts.summarize_history as sh
    with patch.object(sh, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["summarize_history.py"]):
            sh.main()
    out = capsys.readouterr().out
    assert "Alpha" in out


def test_config_package_importable() -> None:
    """The config package exports Settings and constants without errors."""
    from config import COMMIT_TARGET, HISTORY_DIR, SCRIPTS_DIR, Settings
    assert COMMIT_TARGET == 60
    assert HISTORY_DIR.exists()
    assert SCRIPTS_DIR.exists()
    assert callable(Settings)


def test_validate_history_main_skips_non_record_files(tmp_path: Path) -> None:
    """schema.json and commit_schedule.json should not cause validation errors."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "schema.json").write_text("{}")
    (h / "commit_schedule.json").write_text("{}")
    (h / "GoodRepo.json").write_text(json.dumps([{"date": "2026-06-01", "commits": 60}]))
    import scripts.validate_history as vh
    with patch.object(vh, "HISTORY_DIR", h):
        result = vh.main()
    assert result == 0


def test_cleanup_main_removes_old_entries(tmp_path: Path) -> None:
    """cleanup.main() removes entries older than the threshold."""
    h = tmp_path / "history"
    h.mkdir()
    f = h / "OldData.json"
    f.write_text(json.dumps([
        {"date": "2020-01-01", "commits": 5},
        {"date": "2026-06-30", "commits": 60},
    ]))
    import scripts.cleanup as cl
    with patch.object(cl, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["cleanup.py", "--days", "30"]):
            result = cl.main()
    assert result == 0
    kept = json.loads(f.read_text())
    assert len(kept) == 1
    assert kept[0]["date"] == "2026-06-30"


def test_summarize_history_sort_by_commits(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    h = tmp_path / "history"
    h.mkdir()
    (h / "Alpha.json").write_text(json.dumps([{"date": "2026-06-01", "commits": 10}]))
    (h / "Beta.json").write_text(json.dumps([{"date": "2026-06-02", "commits": 99}]))
    import scripts.summarize_history as sh
    with patch.object(sh, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["summarize_history.py", "--sort-by", "commits"]):
            sh.main()
    out = capsys.readouterr().out
    beta_pos = out.index("Beta")
    alpha_pos = out.index("Alpha")
    assert beta_pos < alpha_pos  # higher commits first
