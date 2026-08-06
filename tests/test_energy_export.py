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
