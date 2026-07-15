"""Tests for scripts.foundry_export."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from scripts.foundry_export import (
    DATASET_COLUMNS,
    build_ontology_objects,
    build_run_rows,
    rows_to_csv,
    rows_to_jsonl,
)


def test_build_run_rows_flattens_all_entries(history_dir: Path) -> None:
    rows = build_run_rows(history_dir)
    # SampleRepo has 2 entries, AnotherRepo has 1
    assert len(rows) == 3


def test_build_run_rows_run_key_format(history_dir: Path) -> None:
    rows = build_run_rows(history_dir)
    for row in rows:
        assert row["run_key"] == f"{row['repo']}:{row['date']}"


def test_build_run_rows_sorted_by_repo_then_date(history_dir: Path) -> None:
    rows = build_run_rows(history_dir)
    keys = [(r["repo"], r["date"]) for r in rows]
    assert keys == sorted(keys)


def test_build_run_rows_uses_last_run_fallback(single_entry_history_dir: Path) -> None:
    rows = build_run_rows(single_entry_history_dir)
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-06-20"


def test_build_run_rows_skips_invalid_json(invalid_history_dir: Path) -> None:
    assert build_run_rows(invalid_history_dir) == []


def test_build_run_rows_normalizes_mode_case(multi_repo_history_dir: Path) -> None:
    rows = build_run_rows(multi_repo_history_dir)
    assert all(r["mode"] in ("improvement", "innovation") for r in rows)


def test_build_run_rows_improvement_count(history_dir: Path) -> None:
    rows = build_run_rows(history_dir)
    sample_first = [r for r in rows if r["repo"] == "SampleRepo"][0]
    assert sample_first["improvement_count"] == 2


def test_build_run_rows_non_string_email_status(tmp_path: Path) -> None:
    h = tmp_path / "history"
    h.mkdir()
    (h / "Weird.json").write_text(json.dumps([{"date": "2026-07-01", "commits": 5, "email_status": {"sent": False}}]))
    rows = build_run_rows(h)
    assert rows[0]["email_status"] == ""


def test_rows_to_csv_header_matches_columns(history_dir: Path) -> None:
    text = rows_to_csv(build_run_rows(history_dir))
    header = text.splitlines()[0]
    assert header == ",".join(DATASET_COLUMNS)


def test_rows_to_csv_roundtrip(history_dir: Path) -> None:
    rows = build_run_rows(history_dir)
    parsed = list(csv.DictReader(io.StringIO(rows_to_csv(rows))))
    assert len(parsed) == len(rows)
    assert parsed[0]["repo"] == rows[0]["repo"]


def test_rows_to_csv_empty() -> None:
    text = rows_to_csv([])
    assert text.strip() == ",".join(DATASET_COLUMNS)


def test_rows_to_jsonl_one_line_per_row(history_dir: Path) -> None:
    rows = build_run_rows(history_dir)
    lines = rows_to_jsonl(rows).strip().splitlines()
    assert len(lines) == len(rows)
    assert all(json.loads(line)["run_key"] for line in lines)


def test_rows_to_jsonl_empty() -> None:
    assert rows_to_jsonl([]) == ""


def test_build_ontology_objects_one_repo_object_per_repo(
    multi_repo_history_dir: Path,
) -> None:
    rows = build_run_rows(multi_repo_history_dir)
    objects = build_ontology_objects(rows)
    repo_objects = [o for o in objects if o["objectType"] == "Repository"]
    assert {o["primaryKey"] for o in repo_objects} == {"Alpha", "Beta", "Gamma"}


def test_build_ontology_objects_one_run_object_per_row(
    multi_repo_history_dir: Path,
) -> None:
    rows = build_run_rows(multi_repo_history_dir)
    objects = build_ontology_objects(rows)
    run_objects = [o for o in objects if o["objectType"] == "AutomationRun"]
    assert len(run_objects) == len(rows)


def test_build_ontology_objects_repo_aggregates(multi_repo_history_dir: Path) -> None:
    rows = build_run_rows(multi_repo_history_dir)
    objects = build_ontology_objects(rows)
    alpha = next(o for o in objects if o["objectType"] == "Repository" and o["primaryKey"] == "Alpha")
    assert alpha["properties"]["totalRuns"] == 2
    assert alpha["properties"]["totalCommits"] == 120


def test_build_ontology_objects_empty() -> None:
    assert build_ontology_objects([]) == []


def test_real_history_dir_exports_cleanly() -> None:
    """The repo's own history/ must export without errors."""
    rows = build_run_rows()
    assert len(rows) > 0
    text = rows_to_csv(rows)
    assert text.startswith(",".join(DATASET_COLUMNS))


def test_normalize_mode_lowercases_string() -> None:
    from scripts.foundry_export import _normalize_mode

    assert _normalize_mode("IMPROVEMENT") == "improvement"
    assert _normalize_mode("Innovation") == "innovation"


def test_normalize_mode_non_string_returns_empty() -> None:
    from scripts.foundry_export import _normalize_mode

    assert _normalize_mode(None) == ""
    assert _normalize_mode(42) == ""


def test_rows_to_jsonl_each_line_is_valid_json(multi_repo_history_dir) -> None:
    import json

    from scripts.foundry_export import build_run_rows, rows_to_jsonl

    rows = build_run_rows(multi_repo_history_dir)
    jsonl = rows_to_jsonl(rows)
    for line in jsonl.strip().splitlines():
        parsed = json.loads(line)
        assert "run_key" in parsed


def test_rows_to_jsonl_empty_returns_empty_string() -> None:
    from scripts.foundry_export import rows_to_jsonl

    assert rows_to_jsonl([]) == ""


def test_build_ontology_run_object_has_required_properties(multi_repo_history_dir) -> None:
    from scripts.foundry_export import build_ontology_objects, build_run_rows

    rows = build_run_rows(multi_repo_history_dir)
    objects = build_ontology_objects(rows)
    run_objs = [o for o in objects if o["objectType"] == "AutomationRun"]
    for obj in run_objs:
        props = obj["properties"]
        assert "repository" in props
        assert "runDate" in props
        assert "commits" in props
