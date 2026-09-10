"""Tests for app/energy_export.py."""

from __future__ import annotations

import json

import pytest

from app.energy_export import (
    aggregate_by_hour,
    filter_records,
    kwh_stats_by_building,
    normalize_kwh,
    pivot_by_hour,
    records_to_csv,
    records_to_json,
    split_by_day,
    summarize_export,
    top_buildings_by_kwh,
)

SAMPLE = [
    {"hour": 8, "building_id": "A", "consumption_kwh": 12.0},
    {"hour": 8, "building_id": "B", "consumption_kwh": 8.0},
    {"hour": 14, "building_id": "A", "consumption_kwh": 20.0},
    {"hour": 14, "building_id": "B", "consumption_kwh": 5.0},
]


def test_records_to_csv_has_header() -> None:
    csv_str = records_to_csv(SAMPLE)
    lines = csv_str.strip().splitlines()
    assert "hour" in lines[0]
    assert "consumption_kwh" in lines[0]


def test_records_to_csv_correct_row_count() -> None:
    csv_str = records_to_csv(SAMPLE)
    lines = csv_str.strip().splitlines()
    assert len(lines) == len(SAMPLE) + 1  # header + rows


def test_records_to_csv_empty_returns_empty_string() -> None:
    assert records_to_csv([]) == ""


def test_records_to_json_valid() -> None:
    result = records_to_json(SAMPLE)
    parsed = json.loads(result)
    assert len(parsed) == len(SAMPLE)


def test_records_to_json_indent() -> None:
    result = records_to_json(SAMPLE, indent=4)
    assert "    " in result


def test_filter_records_by_min_kwh() -> None:
    result = filter_records(SAMPLE, min_kwh=10.0)
    assert all(r["consumption_kwh"] >= 10.0 for r in result)


def test_filter_records_by_max_kwh() -> None:
    result = filter_records(SAMPLE, max_kwh=10.0)
    assert all(r["consumption_kwh"] <= 10.0 for r in result)


def test_filter_records_by_hour() -> None:
    result = filter_records(SAMPLE, hour=8)
    assert all(r["hour"] == 8 for r in result)
    assert len(result) == 2


def test_filter_records_by_building_id() -> None:
    result = filter_records(SAMPLE, building_id="A")
    assert all(r["building_id"] == "A" for r in result)
    assert len(result) == 2


def test_filter_records_combined() -> None:
    result = filter_records(SAMPLE, hour=14, building_id="A")
    assert len(result) == 1
    assert result[0]["consumption_kwh"] == 20.0


def test_aggregate_by_hour_groups_correctly() -> None:
    result = aggregate_by_hour(SAMPLE)
    hours = {r["hour"] for r in result}
    assert hours == {8, 14}


def test_aggregate_by_hour_mean_correct() -> None:
    result = aggregate_by_hour(SAMPLE)
    hour8 = next(r for r in result if r["hour"] == 8)
    assert hour8["mean_kwh"] == pytest.approx(10.0)


def test_aggregate_by_hour_empty() -> None:
    result = aggregate_by_hour([])
    assert result == []


def test_summarize_export_total_records() -> None:
    s = summarize_export(SAMPLE)
    assert s["total_records"] == 4


def test_summarize_export_total_kwh() -> None:
    s = summarize_export(SAMPLE)
    assert s["total_kwh"] == pytest.approx(45.0)


def test_summarize_export_empty() -> None:
    s = summarize_export([])
    assert s["total_records"] == 0
    assert s["mean_kwh"] == 0.0
    assert s["min_kwh"] is None


@pytest.mark.parametrize("min_kwh,expected_count", [(0.0, 4), (10.0, 2), (25.0, 0)])
def test_filter_min_kwh_parametrize(min_kwh, expected_count) -> None:
    result = filter_records(SAMPLE, min_kwh=min_kwh)
    assert len(result) == expected_count


@pytest.mark.parametrize("max_kwh,expected_count", [(100.0, 4), (10.0, 2), (0.0, 0)])
def test_filter_max_kwh_parametrize(max_kwh, expected_count) -> None:
    result = filter_records(SAMPLE, max_kwh=max_kwh)
    assert len(result) == expected_count


def test_records_to_json_empty() -> None:
    result = records_to_json([])
    assert result == "[]"


def test_aggregate_by_hour_counts() -> None:
    result = aggregate_by_hour(SAMPLE)
    assert all("count" in r for r in result)
    hour8 = next(r for r in result if r["hour"] == 8)
    assert hour8["count"] == 2


def test_summarize_export_min_max() -> None:
    s = summarize_export(SAMPLE)
    assert s["min_kwh"] == pytest.approx(5.0)
    assert s["max_kwh"] == pytest.approx(20.0)


@pytest.mark.parametrize("hour,expected_count", [(8, 2), (14, 2), (0, 0)])
def test_filter_by_hour_parametrize(hour, expected_count) -> None:
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


class TestCountRecordsByField:
    def test_basic(self) -> None:
        from app.energy_export import count_records_by_field

        records = [{"type": "A"}, {"type": "B"}, {"type": "A"}]
        result = count_records_by_field(records, "type")
        assert result["A"] == 2
        assert result["B"] == 1

    def test_empty(self) -> None:
        from app.energy_export import count_records_by_field

        assert count_records_by_field([], "type") == {}

    def test_missing_field(self) -> None:
        from app.energy_export import count_records_by_field

        result = count_records_by_field([{"x": 1}], "type")
        assert "" in result


class TestRecordsToTsv:
    def test_basic(self) -> None:
        from app.energy_export import records_to_tsv

        records = [{"a": 1, "b": 2}]
        result = records_to_tsv(records, columns=["a", "b"])
        assert "\t" in result
        assert "a\tb" in result

    def test_empty(self) -> None:
        from app.energy_export import records_to_tsv

        assert records_to_tsv([]) == ""

    def test_two_rows(self) -> None:
        from app.energy_export import records_to_tsv

        records = [{"x": 1}, {"x": 2}]
        lines = records_to_tsv(records, columns=["x"]).split("\n")
        assert len(lines) == 3


class TestMergeRecords:
    def test_override_replaces(self) -> None:
        from app.energy_export import merge_records

        base = [{"id": "1", "v": 10}]
        override = [{"id": "1", "v": 99}]
        result = merge_records(base, override, key="id")
        assert len(result) == 1
        assert result[0]["v"] == 99

    def test_adds_new(self) -> None:
        from app.energy_export import merge_records

        base = [{"id": "1", "v": 10}]
        override = [{"id": "2", "v": 20}]
        result = merge_records(base, override, key="id")
        assert len(result) == 2


class TestSampleRecords:
    def test_size(self) -> None:
        from app.energy_export import sample_records

        records = [{"i": i} for i in range(20)]
        result = sample_records(records, n=5)
        assert len(result) == 5

    def test_reproducible(self) -> None:
        from app.energy_export import sample_records

        records = [{"i": i} for i in range(50)]
        a = sample_records(records, n=10, seed=7)
        b = sample_records(records, n=10, seed=7)
        assert a == b

    def test_negative_n_raises(self) -> None:
        from app.energy_export import sample_records

        with pytest.raises(ValueError):
            sample_records([{"x": 1}], n=-1)

    def test_n_larger_than_records(self) -> None:
        from app.energy_export import sample_records

        records = [{"i": i} for i in range(3)]
        result = sample_records(records, n=10)
        assert len(result) == 3


def test_top_buildings_by_kwh_order() -> None:
    result = top_buildings_by_kwh(SAMPLE, n=2)
    assert result[0]["building_id"] == "A"
    assert result[0]["total_kwh"] == pytest.approx(32.0)


def test_top_buildings_by_kwh_n_limit() -> None:
    result = top_buildings_by_kwh(SAMPLE, n=1)
    assert len(result) == 1


def test_top_buildings_by_kwh_empty() -> None:
    result = top_buildings_by_kwh([])
    assert result == []


def test_pivot_by_hour_structure() -> None:
    pivot = pivot_by_hour(SAMPLE)
    assert "A" in pivot
    assert "B" in pivot
    assert 8 in pivot["A"]
    assert pivot["A"][8] == pytest.approx(12.0)


def test_pivot_by_hour_accumulates() -> None:
    records = [
        {"hour": 8, "building_id": "A", "consumption_kwh": 5.0},
        {"hour": 8, "building_id": "A", "consumption_kwh": 3.0},
    ]
    pivot = pivot_by_hour(records)
    assert pivot["A"][8] == pytest.approx(8.0)


def test_pivot_by_hour_empty() -> None:
    assert pivot_by_hour([]) == {}


def test_normalize_kwh_range() -> None:
    result = normalize_kwh(SAMPLE)
    kwh_vals = [r["consumption_kwh"] for r in result]
    assert min(kwh_vals) == pytest.approx(0.0)
    assert max(kwh_vals) == pytest.approx(1.0)


def test_normalize_kwh_preserves_other_fields() -> None:
    result = normalize_kwh(SAMPLE)
    assert result[0]["building_id"] == SAMPLE[0]["building_id"]


def test_normalize_kwh_empty() -> None:
    result = normalize_kwh([])
    assert result == []


@pytest.mark.parametrize("n,expected_len", [(1, 1), (2, 2), (10, 2)])
def test_top_buildings_n_parametrize(n, expected_len) -> None:
    result = top_buildings_by_kwh(SAMPLE, n=n)
    assert len(result) == expected_len


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


def test_flatten_nested_records_basic() -> None:
    from app.energy_export import flatten_nested_records

    records = [{"a": {"b": 1, "c": 2}, "d": 3}]
    result = flatten_nested_records(records)
    assert "d" in result[0]
    assert result[0]["d"] == 3


def test_flatten_nested_records_empty_input() -> None:
    from app.energy_export import flatten_nested_records

    assert flatten_nested_records([]) == []


def test_flatten_nested_records_no_nesting() -> None:
    from app.energy_export import flatten_nested_records

    records = [{"x": 1, "y": 2}]
    result = flatten_nested_records(records)
    assert result[0]["x"] == 1
    assert result[0]["y"] == 2


@pytest.mark.parametrize("n_records", [1, 5, 10])
def test_flatten_nested_records_preserves_count(n_records: int) -> None:
    from app.energy_export import flatten_nested_records

    records = [{"v": i} for i in range(n_records)]
    result = flatten_nested_records(records)
    assert len(result) == n_records


class TestRecordsMissingFields:
    def test_all_complete_returns_empty(self) -> None:
        from app.energy_export import records_missing_fields

        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert records_missing_fields(records, ["a", "b"]) == []

    def test_missing_field_detected(self) -> None:
        from app.energy_export import records_missing_fields

        records = [{"a": 1}, {"a": 2, "b": 3}]
        assert records_missing_fields(records, ["a", "b"]) == [0]

    def test_none_value_treated_as_missing(self) -> None:
        from app.energy_export import records_missing_fields

        records = [{"a": None, "b": 2}]
        assert records_missing_fields(records, ["a"]) == [0]

    @pytest.mark.parametrize(
        "records,required,expected",
        [
            ([{"x": 1}], ["x"], []),
            ([{"x": 1}, {}], ["x"], [1]),
        ],
    )
    def test_parametrized(self, records, required, expected) -> None:
        from app.energy_export import records_missing_fields

        assert records_missing_fields(records, required) == expected


class TestRecordsToLookup:
    def test_basic_lookup(self) -> None:
        from app.energy_export import records_to_lookup

        records = [{"id": "a", "val": 1}, {"id": "b", "val": 2}]
        lookup = records_to_lookup(records, "id")
        assert lookup["a"]["val"] == 1
        assert lookup["b"]["val"] == 2

    def test_last_wins_on_duplicate(self) -> None:
        from app.energy_export import records_to_lookup

        records = [{"id": "a", "val": 1}, {"id": "a", "val": 2}]
        lookup = records_to_lookup(records, "id")
        assert lookup["a"]["val"] == 2

    def test_missing_key_raises(self) -> None:
        from app.energy_export import records_to_lookup

        with pytest.raises(KeyError):
            records_to_lookup([{"x": 1}], "id")


class TestRenameRecordFields:
    def test_field_renamed(self) -> None:
        from app.energy_export import rename_record_fields

        records = [{"old_name": 1}]
        result = rename_record_fields(records, {"old_name": "new_name"})
        assert "new_name" in result[0]
        assert "old_name" not in result[0]

    def test_unmapped_fields_unchanged(self) -> None:
        from app.energy_export import rename_record_fields

        records = [{"a": 1, "b": 2}]
        result = rename_record_fields(records, {"a": "x"})
        assert "b" in result[0]

    def test_empty_mapping_returns_copy(self) -> None:
        from app.energy_export import rename_record_fields

        records = [{"k": "v"}]
        result = rename_record_fields(records, {})
        assert result[0] == {"k": "v"}

    def test_original_not_mutated(self) -> None:
        from app.energy_export import rename_record_fields

        records = [{"old": 1}]
        rename_record_fields(records, {"old": "new"})
        assert "old" in records[0]


class TestPivotByHour:
    def test_basic_pivot(self) -> None:
        from app.energy_export import pivot_by_hour

        records = [
            {"label": "A", "hour": 9, "value": 100.0},
            {"label": "A", "hour": 10, "value": 120.0},
            {"label": "B", "hour": 9, "value": 80.0},
        ]
        result = pivot_by_hour(records, "hour", "value", "label")
        assert result["A"][9] == 100.0
        assert result["B"][9] == 80.0

    def test_empty_returns_empty(self) -> None:
        from app.energy_export import pivot_by_hour

        assert pivot_by_hour([], "hour", "value", "label") == {}


class TestFlatToWide:
    def test_basic_pivot(self) -> None:
        from app.energy_export import flat_to_wide

        records = [
            {"id": "S1", "key": "temp", "value": 21.0},
            {"id": "S1", "key": "humidity", "value": 55.0},
        ]
        result = flat_to_wide(records, "id", "key", "value")
        assert result["S1"]["temp"] == 21.0
        assert result["S1"]["humidity"] == 55.0

    def test_empty_returns_empty(self) -> None:
        from app.energy_export import flat_to_wide

        assert flat_to_wide([], "id", "key", "value") == {}


class TestFilterRecordsByValue:
    def test_basic_filter(self) -> None:
        from app.energy_export import filter_records_by_value

        records = [{"v": 5.0}, {"v": 15.0}, {"v": 25.0}]
        result = filter_records_by_value(records, "v", 10.0, 20.0)
        assert len(result) == 1
        assert result[0]["v"] == 15.0

    def test_inclusive_bounds(self) -> None:
        from app.energy_export import filter_records_by_value

        records = [{"v": 0.0}, {"v": 10.0}]
        result = filter_records_by_value(records, "v", 0.0, 10.0)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        from app.energy_export import filter_records_by_value

        assert filter_records_by_value([], "v", 0.0, 10.0) == []


class TestKwhToMwh:
    def test_basic(self) -> None:
        from app.energy_export import kwh_to_mwh

        assert kwh_to_mwh(5000.0) == pytest.approx(5.0)

    def test_zero(self) -> None:
        from app.energy_export import kwh_to_mwh

        assert kwh_to_mwh(0.0) == pytest.approx(0.0)

    def test_negative_raises(self) -> None:
        from app.energy_export import kwh_to_mwh

        with pytest.raises(ValueError):
            kwh_to_mwh(-1.0)


class TestMwhToKwh:
    def test_basic(self) -> None:
        from app.energy_export import mwh_to_kwh

        assert mwh_to_kwh(5.0) == pytest.approx(5000.0)

    def test_zero(self) -> None:
        from app.energy_export import mwh_to_kwh

        assert mwh_to_kwh(0.0) == pytest.approx(0.0)

    def test_negative_raises(self) -> None:
        from app.energy_export import mwh_to_kwh

        with pytest.raises(ValueError):
            mwh_to_kwh(-1.0)


class TestDailyPeakConsumption:
    def test_finds_max(self) -> None:
        from app.energy_export import daily_peak_consumption

        profile = [float(i) for i in range(24)]
        assert daily_peak_consumption(profile) == pytest.approx(23.0)

    def test_wrong_length_raises(self) -> None:
        from app.energy_export import daily_peak_consumption

        with pytest.raises(ValueError):
            daily_peak_consumption([1.0, 2.0])


class TestRecordsTotalKwh:
    def test_sums_kwh_values(self) -> None:
        from app.energy_export import records_total_kwh

        assert records_total_kwh(SAMPLE) == pytest.approx(45.0)

    def test_empty_returns_zero(self) -> None:
        from app.energy_export import records_total_kwh

        assert records_total_kwh([]) == 0.0

    def test_records_without_kwh_skipped(self) -> None:
        from app.energy_export import records_total_kwh

        assert records_total_kwh([{"hour": 0}, {"consumption_kwh": 5.0}]) == pytest.approx(5.0)


class TestRecordsKwhRange:
    def test_returns_min_max(self) -> None:
        from app.energy_export import records_kwh_range

        lo, hi = records_kwh_range(SAMPLE)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(20.0)

    def test_no_kwh_field_returns_none(self) -> None:
        from app.energy_export import records_kwh_range

        assert records_kwh_range([{"hour": 0}]) is None

    def test_empty_returns_none(self) -> None:
        from app.energy_export import records_kwh_range

        assert records_kwh_range([]) is None
