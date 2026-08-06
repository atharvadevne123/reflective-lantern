"""Tests for app/energy_export.py."""

from __future__ import annotations

import json

import pytest

from app.energy_export import (
    aggregate_by_hour,
    filter_records,
    normalize_kwh,
    pivot_by_hour,
    records_to_csv,
    records_to_json,
    summarize_export,
    top_buildings_by_kwh,
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


def test_top_buildings_by_kwh_order():
    result = top_buildings_by_kwh(SAMPLE, n=2)
    assert result[0]["building_id"] == "A"
    assert result[0]["total_kwh"] == pytest.approx(32.0)


def test_top_buildings_by_kwh_n_limit():
    result = top_buildings_by_kwh(SAMPLE, n=1)
    assert len(result) == 1


def test_top_buildings_by_kwh_empty():
    result = top_buildings_by_kwh([])
    assert result == []


def test_pivot_by_hour_structure():
    pivot = pivot_by_hour(SAMPLE)
    assert "A" in pivot
    assert "B" in pivot
    assert 8 in pivot["A"]
    assert pivot["A"][8] == pytest.approx(12.0)


def test_pivot_by_hour_accumulates():
    records = [
        {"hour": 8, "building_id": "A", "consumption_kwh": 5.0},
        {"hour": 8, "building_id": "A", "consumption_kwh": 3.0},
    ]
    pivot = pivot_by_hour(records)
    assert pivot["A"][8] == pytest.approx(8.0)


def test_pivot_by_hour_empty():
    assert pivot_by_hour([]) == {}


def test_normalize_kwh_range():
    result = normalize_kwh(SAMPLE)
    kwh_vals = [r["consumption_kwh"] for r in result]
    assert min(kwh_vals) == pytest.approx(0.0)
    assert max(kwh_vals) == pytest.approx(1.0)


def test_normalize_kwh_preserves_other_fields():
    result = normalize_kwh(SAMPLE)
    assert result[0]["building_id"] == SAMPLE[0]["building_id"]


def test_normalize_kwh_empty():
    result = normalize_kwh([])
    assert result == []


@pytest.mark.parametrize("n,expected_len", [(1, 1), (2, 2), (10, 2)])
def test_top_buildings_n_parametrize(n, expected_len):
    result = top_buildings_by_kwh(SAMPLE, n=n)
    assert len(result) == expected_len


# Tests for split_by_day and kwh_stats_by_building
from app.energy_export import kwh_stats_by_building, split_by_day


def test_split_by_day_basic() -> None:
    records = [
        {"timestamp": "2024-06-01T10:00:00", "v": 1},
        {"timestamp": "2024-06-01T14:00:00", "v": 2},
        {"timestamp": "2024-06-02T08:00:00", "v": 3},
    ]
    buckets = split_by_day(records)
    assert "2024-06-01" in buckets
    assert "2024-06-02" in buckets
    assert len(buckets["2024-06-01"]) == 2
    assert len(buckets["2024-06-02"]) == 1


def test_split_by_day_missing_field_goes_to_unknown() -> None:
    records = [{"v": 1}]
    buckets = split_by_day(records)
    assert "_unknown" in buckets


def test_split_by_day_empty() -> None:
    assert split_by_day([]) == {}


def test_kwh_stats_by_building_basic() -> None:
    records = [
        {"building_id": "A", "consumption_kwh": 10.0},
        {"building_id": "A", "consumption_kwh": 20.0},
        {"building_id": "B", "consumption_kwh": 15.0},
    ]
    result = kwh_stats_by_building(records)
    assert "A" in result
    assert "B" in result
    assert result["A"]["total"] == pytest.approx(30.0)
    assert result["A"]["count"] == 2.0
    assert result["B"]["mean"] == pytest.approx(15.0)


def test_kwh_stats_by_building_missing_fields_skipped() -> None:
    records = [
        {"building_id": "A"},
        {"consumption_kwh": 10.0},
        {},
    ]
    result = kwh_stats_by_building(records)
    assert result == {}


def test_kwh_stats_by_building_empty() -> None:
    assert kwh_stats_by_building([]) == {}


def test_kwh_stats_by_building_keys() -> None:
    records = [{"building_id": "X", "consumption_kwh": 5.0}]
    stats = kwh_stats_by_building(records)
    assert "total" in stats["X"]
    assert "mean" in stats["X"]
    assert "min" in stats["X"]
    assert "max" in stats["X"]
    assert "count" in stats["X"]


@pytest.mark.parametrize("n", [1, 3, 5])
def test_split_by_day_single_day_count(n: int) -> None:
    records = [{"timestamp": "2024-01-15T00:00:00", "v": i} for i in range(n)]
    buckets = split_by_day(records)
    assert len(buckets["2024-01-15"]) == n
