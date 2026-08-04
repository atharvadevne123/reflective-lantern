"""Tests for app/energy_export.py."""

from __future__ import annotations

import json

import pytest

from app.energy_export import (
    aggregate_by_hour,
    filter_records,
    records_to_csv,
    records_to_json,
    summarize_export,
)

SAMPLE = [
    {"hour": 8, "building_id": "A", "consumption_kwh": 12.0},
    {"hour": 8, "building_id": "B", "consumption_kwh": 8.0},
    {"hour": 14, "building_id": "A", "consumption_kwh": 20.0},
    {"hour": 14, "building_id": "B", "consumption_kwh": 5.0},
]


def test_records_to_csv_has_header():
    csv_str = records_to_csv(SAMPLE)
    lines = csv_str.strip().splitlines()
    assert "hour" in lines[0]
    assert "consumption_kwh" in lines[0]


def test_records_to_csv_correct_row_count():
    csv_str = records_to_csv(SAMPLE)
    lines = csv_str.strip().splitlines()
    assert len(lines) == len(SAMPLE) + 1  # header + rows


def test_records_to_csv_empty_returns_empty_string():
    assert records_to_csv([]) == ""


def test_records_to_json_valid():
    result = records_to_json(SAMPLE)
    parsed = json.loads(result)
    assert len(parsed) == len(SAMPLE)


def test_records_to_json_indent():
    result = records_to_json(SAMPLE, indent=4)
    assert "    " in result


def test_filter_records_by_min_kwh():
    result = filter_records(SAMPLE, min_kwh=10.0)
    assert all(r["consumption_kwh"] >= 10.0 for r in result)


def test_filter_records_by_max_kwh():
    result = filter_records(SAMPLE, max_kwh=10.0)
    assert all(r["consumption_kwh"] <= 10.0 for r in result)


def test_filter_records_by_hour():
    result = filter_records(SAMPLE, hour=8)
    assert all(r["hour"] == 8 for r in result)
    assert len(result) == 2


def test_filter_records_by_building_id():
    result = filter_records(SAMPLE, building_id="A")
    assert all(r["building_id"] == "A" for r in result)
    assert len(result) == 2


def test_filter_records_combined():
    result = filter_records(SAMPLE, hour=14, building_id="A")
    assert len(result) == 1
    assert result[0]["consumption_kwh"] == 20.0


def test_aggregate_by_hour_groups_correctly():
    result = aggregate_by_hour(SAMPLE)
    hours = {r["hour"] for r in result}
    assert hours == {8, 14}


def test_aggregate_by_hour_mean_correct():
    result = aggregate_by_hour(SAMPLE)
    hour8 = next(r for r in result if r["hour"] == 8)
    assert hour8["mean_kwh"] == pytest.approx(10.0)


def test_aggregate_by_hour_empty():
    result = aggregate_by_hour([])
    assert result == []


def test_summarize_export_total_records():
    s = summarize_export(SAMPLE)
    assert s["total_records"] == 4


def test_summarize_export_total_kwh():
    s = summarize_export(SAMPLE)
    assert s["total_kwh"] == pytest.approx(45.0)


def test_summarize_export_empty():
    s = summarize_export([])
    assert s["total_records"] == 0
    assert s["mean_kwh"] == 0.0
    assert s["min_kwh"] is None


@pytest.mark.parametrize("min_kwh,expected_count", [(0.0, 4), (10.0, 2), (25.0, 0)])
def test_filter_min_kwh_parametrize(min_kwh, expected_count):
    result = filter_records(SAMPLE, min_kwh=min_kwh)
    assert len(result) == expected_count


@pytest.mark.parametrize("max_kwh,expected_count", [(100.0, 4), (10.0, 2), (0.0, 0)])
def test_filter_max_kwh_parametrize(max_kwh, expected_count):
    result = filter_records(SAMPLE, max_kwh=max_kwh)
    assert len(result) == expected_count


def test_records_to_json_empty():
    result = records_to_json([])
    assert result == "[]"


def test_aggregate_by_hour_counts():
    result = aggregate_by_hour(SAMPLE)
    assert all("count" in r for r in result)
    hour8 = next(r for r in result if r["hour"] == 8)
    assert hour8["count"] == 2


def test_summarize_export_min_max():
    s = summarize_export(SAMPLE)
    assert s["min_kwh"] == pytest.approx(5.0)
    assert s["max_kwh"] == pytest.approx(20.0)


@pytest.mark.parametrize("hour,expected_count", [(8, 2), (14, 2), (0, 0)])
def test_filter_by_hour_parametrize(hour, expected_count):
    result = filter_records(SAMPLE, hour=hour)
    assert len(result) == expected_count


class TestRecordsToJsonl:
    def test_each_line_valid_json(self) -> None:
        import json

        from app.energy_export import records_to_jsonl

        out = records_to_jsonl(SAMPLE)
        lines = [ln for ln in out.strip().split("\n") if ln]
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_line_count_matches_records(self) -> None:
        from app.energy_export import records_to_jsonl

        out = records_to_jsonl(SAMPLE)
        lines = [ln for ln in out.strip().split("\n") if ln]
        assert len(lines) == len(SAMPLE)

    def test_empty_records_returns_empty_string(self) -> None:
        from app.energy_export import records_to_jsonl

        assert records_to_jsonl([]) == ""

    def test_output_ends_with_newline(self) -> None:
        from app.energy_export import records_to_jsonl

        out = records_to_jsonl(SAMPLE[:1])
        assert out.endswith("\n")


class TestDeduplicateRecords:
    def test_removes_exact_duplicates(self) -> None:
        from app.energy_export import deduplicate_records

        records = [
            {"building_id": "A", "timestamp": "2024-01-01", "kwh": 5.0},
            {"building_id": "A", "timestamp": "2024-01-01", "kwh": 5.0},
            {"building_id": "B", "timestamp": "2024-01-01", "kwh": 3.0},
        ]
        result = deduplicate_records(records)
        assert len(result) == 2

    def test_preserves_first_occurrence(self) -> None:
        from app.energy_export import deduplicate_records

        records = [
            {"building_id": "A", "timestamp": "T1", "kwh": 1.0},
            {"building_id": "A", "timestamp": "T1", "kwh": 2.0},
        ]
        result = deduplicate_records(records)
        assert result[0]["kwh"] == 1.0

    def test_custom_key_fields(self) -> None:
        from app.energy_export import deduplicate_records

        records = [
            {"meter_id": "M1", "hour": 8, "kwh": 4.0},
            {"meter_id": "M1", "hour": 8, "kwh": 5.0},
            {"meter_id": "M2", "hour": 8, "kwh": 3.0},
        ]
        result = deduplicate_records(records, key_fields=["meter_id", "hour"])
        assert len(result) == 2

    def test_no_duplicates_unchanged(self) -> None:
        from app.energy_export import deduplicate_records

        records = [{"building_id": str(i), "timestamp": "T1"} for i in range(5)]
        result = deduplicate_records(records)
        assert len(result) == 5

    @pytest.mark.parametrize("n_dupes", [1, 5, 10])
    def test_single_unique_key_collapses_all(self, n_dupes: int) -> None:
        from app.energy_export import deduplicate_records

        records = [{"building_id": "X", "timestamp": "T"} for _ in range(n_dupes)]
        result = deduplicate_records(records)
        assert len(result) == 1


class TestSortRecords:
    def test_sorts_ascending(self) -> None:
        from app.energy_export import sort_records

        records = [{"ts": "2026-08-03"}, {"ts": "2026-08-01"}, {"ts": "2026-08-02"}]
        result = sort_records(records, key="ts")
        assert result[0]["ts"] == "2026-08-01"

    def test_sorts_descending(self) -> None:
        from app.energy_export import sort_records

        records = [{"ts": "2026-08-01"}, {"ts": "2026-08-03"}]
        result = sort_records(records, key="ts", reverse=True)
        assert result[0]["ts"] == "2026-08-03"

    def test_does_not_mutate_input(self) -> None:
        from app.energy_export import sort_records

        records = [{"ts": "b"}, {"ts": "a"}]
        original = list(records)
        sort_records(records, key="ts")
        assert records == original

    def test_empty(self) -> None:
        from app.energy_export import sort_records

        assert sort_records([], key="ts") == []


class TestPartitionRecords:
    def test_splits_by_value(self) -> None:
        from app.energy_export import partition_records

        records = [
            {"type": "A"},
            {"type": "B"},
            {"type": "A"},
        ]
        matches, non_matches = partition_records(records, "type", "A")
        assert len(matches) == 2
        assert len(non_matches) == 1

    def test_no_matches(self) -> None:
        from app.energy_export import partition_records

        records = [{"type": "X"}]
        matches, non_matches = partition_records(records, "type", "Y")
        assert matches == []
        assert len(non_matches) == 1

    def test_all_match(self) -> None:
        from app.energy_export import partition_records

        records = [{"k": 1}, {"k": 1}]
        matches, non_matches = partition_records(records, "k", 1)
        assert len(matches) == 2
        assert non_matches == []
