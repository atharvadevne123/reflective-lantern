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
    (h / "Alpha.json").write_text(
        json.dumps(
            [
                {"date": "2026-06-30", "mode": "improvement", "commits": 60, "tests_passed": True},
                {"date": "2026-06-15", "mode": "improvement", "commits": 55, "tests_passed": False},
            ]
        )
    )
    (h / "Beta.json").write_text(json.dumps({"last_run": "2026-06-20", "commits": 30}))
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
def test_main_runs_without_error(
    history_path: Path, sort_by: str, capsys: pytest.CaptureFixture
) -> None:
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


@pytest.mark.parametrize(
    "commits,expected_commits",
    [
        (60, 60),
        (0, 0),
        (120, 120),
    ],
)
def test_load_latest_entry_commit_values(
    tmp_path: Path, commits: int, expected_commits: int
) -> None:
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
    (h / "commit_schedule.json").write_text(json.dumps({"start_year": 2026, "start_week": 1}))
    import sys

    with (
        patch.object(sh, "HISTORY_DIR", h),
        patch.object(sys, "argv", ["sh.py"]),
    ):
        sh.main()
    out = capsys.readouterr().out
    assert "MyRepo" in out
    assert "commit_schedule" not in out


def test_main_json_flag_outputs_valid_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import sys
    from unittest.mock import patch

    import scripts.summarize_history as sh

    h = tmp_path / "history"
    h.mkdir()
    (h / "AlphaRepo.json").write_text(json.dumps([{"date": "2026-06-30", "commits": 55}]))
    with patch.object(sh, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["summarize_history.py", "--json"]):
            sh.main()
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0]["repo"] == "AlphaRepo"
    assert parsed[0]["commits"] == 55


def test_main_sort_by_repo_alphabetical(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import sys
    from unittest.mock import patch

    import scripts.summarize_history as sh

    h = tmp_path / "history"
    h.mkdir()
    (h / "Zebra.json").write_text(json.dumps([{"date": "2026-06-10", "commits": 5}]))
    (h / "Apple.json").write_text(json.dumps([{"date": "2026-06-11", "commits": 10}]))
    with patch.object(sh, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["summarize_history.py", "--sort-by", "repo"]):
            sh.main()
    out = capsys.readouterr().out
    assert out.index("Apple") < out.index("Zebra")


def test_main_json_tests_passed_field(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import sys
    from unittest.mock import patch

    import scripts.summarize_history as sh

    h = tmp_path / "history"
    h.mkdir()
    (h / "Repo.json").write_text(
        json.dumps([{"date": "2026-07-01", "commits": 60, "tests_passed": True}])
    )
    with patch.object(sh, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["summarize_history.py", "--json"]):
            sh.main()
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed[0]["tests_passed"] is True


def test_filter_mode_improvement(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import sys
    from unittest.mock import patch

    import scripts.summarize_history as sh

    h = tmp_path / "history"
    h.mkdir()
    (h / "A.json").write_text(
        json.dumps([{"date": "2026-07-01", "commits": 60, "mode": "improvement"}])
    )
    (h / "B.json").write_text(
        json.dumps([{"date": "2026-07-08", "commits": 60, "mode": "innovation"}])
    )
    with patch.object(sh, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["summarize_history.py", "--filter-mode", "improvement"]):
            sh.main()
    out = capsys.readouterr().out
    assert "A" in out
    assert "B" not in out.split("1 repos")[0]


def test_filter_mode_innovation(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import sys
    from unittest.mock import patch

    import scripts.summarize_history as sh

    h = tmp_path / "history"
    h.mkdir()
    (h / "A.json").write_text(
        json.dumps([{"date": "2026-07-01", "commits": 60, "mode": "improvement"}])
    )
    (h / "B.json").write_text(
        json.dumps([{"date": "2026-07-08", "commits": 60, "mode": "innovation"}])
    )
    with patch.object(sh, "HISTORY_DIR", h):
        with patch.object(sys, "argv", ["summarize_history.py", "--filter-mode", "innovation"]):
            sh.main()
    out = capsys.readouterr().out
    assert "B" in out
    assert "1 repos" in out


def test_load_all_entries_list_file(tmp_path: Path) -> None:
    import scripts.summarize_history as sh

    f = tmp_path / "repo.json"
    f.write_text(
        json.dumps(
            [
                {"date": "2026-07-01", "commits": 60},
                {"date": "2026-07-02", "commits": 45},
            ]
        )
    )
    result = sh.load_all_entries(f)
    assert len(result) == 2


def test_load_all_entries_dict_file(tmp_path: Path) -> None:
    import scripts.summarize_history as sh

    f = tmp_path / "repo.json"
    f.write_text(json.dumps({"date": "2026-07-01", "commits": 60}))
    result = sh.load_all_entries(f)
    assert len(result) == 1


def test_load_all_entries_invalid_json(tmp_path: Path) -> None:
    import scripts.summarize_history as sh

    f = tmp_path / "bad.json"
    f.write_text("{bad json")
    result = sh.load_all_entries(f)
    assert result == []


def test_load_all_entries_empty_list(tmp_path: Path) -> None:
    import scripts.summarize_history as sh

    f = tmp_path / "empty.json"
    f.write_text("[]")
    result = sh.load_all_entries(f)
    assert result == []


def test_load_all_entries_filters_non_dicts(tmp_path: Path) -> None:
    import scripts.summarize_history as sh

    f = tmp_path / "mixed.json"
    f.write_text(json.dumps([{"date": "2026-07-01", "commits": 60}, "not-a-dict", 42]))
    result = sh.load_all_entries(f)
    assert len(result) == 1


def test_aggregate_stats_total_commits(history_dir: Path) -> None:
    import scripts.summarize_history as sh

    entries = []
    for path in history_dir.glob("*.json"):
        entries.extend(sh.load_all_entries(path))
    total_commits = sum(e.get("commits", 0) for e in entries)
    assert total_commits > 0


def test_summarize_history_returns_zero(history_dir: Path) -> None:
    import sys
    from unittest.mock import patch

    import scripts.summarize_history as sh

    with patch.object(sys, "argv", ["summarize_history.py"]):
        with patch.object(sh, "HISTORY_DIR", history_dir):
            rc = sh.main()
    assert rc in (0, None)


def test_export_to_csv_creates_file(history_dir: Path, tmp_path: Path) -> None:
    from scripts.summarize_history import export_to_csv

    out = tmp_path / "summary.csv"
    count = export_to_csv(history_dir=history_dir, output_path=out)
    assert out.exists()
    assert count >= 0


def test_export_to_csv_header_row(history_dir: Path, tmp_path: Path) -> None:
    import csv
    from scripts.summarize_history import export_to_csv

    out = tmp_path / "summary.csv"
    export_to_csv(history_dir=history_dir, output_path=out)
    if out.exists():
        with out.open() as fh:
            reader = csv.DictReader(fh)
            assert "repo" in (reader.fieldnames or [])


def test_export_to_csv_empty_history_returns_zero(tmp_path: Path) -> None:
    from scripts.summarize_history import export_to_csv

    empty_dir = tmp_path / "history"
    empty_dir.mkdir()
    out = tmp_path / "out.csv"
    count = export_to_csv(history_dir=empty_dir, output_path=out)
    assert count == 0
    assert not out.exists()
