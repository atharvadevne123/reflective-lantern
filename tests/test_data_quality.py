"""Tests for app/data_quality.py."""

from __future__ import annotations

import pytest

from app.data_quality import (
    batch_score,
    completeness_score,
    detect_duplicates,
    field_type_consistency,
    fill_missing,
    flag_outliers,
    quality_summary,
    range_violation_count,
    score_record,
)

GOOD_RECORD = {
    "hour": 10,
    "month": 6,
    "day_of_week": 2,
    "consumption_kwh": 15.0,
    "temperature_c": 22.0,
    "humidity_pct": 55.0,
}


def test_perfect_record_scores_100() -> None:
    result = score_record(GOOD_RECORD)
    assert result["dq_score"] == 100
    assert result["dq_issues"] == []


def test_missing_hour_deducts_points() -> None:
    rec = {**GOOD_RECORD}
    del rec["hour"]
    result = score_record(rec)
    assert result["dq_score"] < 100
    assert any("missing:hour" in i for i in result["dq_issues"])


def test_invalid_hour_deducts_points() -> None:
    rec = {**GOOD_RECORD, "hour": 99}
    result = score_record(rec)
    assert result["dq_score"] < 100
    assert any("invalid:hour" in i for i in result["dq_issues"])


def test_negative_kwh_deducts_points() -> None:
    rec = {**GOOD_RECORD, "consumption_kwh": -5.0}
    result = score_record(rec)
    assert result["dq_score"] < 100
    assert any("negative:consumption_kwh" in i for i in result["dq_issues"])


def test_extreme_kwh_deducts_minor_points() -> None:
    rec = {**GOOD_RECORD, "consumption_kwh": 9999.0}
    result = score_record(rec)
    assert result["dq_score"] < 100
    assert any("extreme:consumption_kwh" in i for i in result["dq_issues"])


def test_invalid_humidity_deducts_points() -> None:
    rec = {**GOOD_RECORD, "humidity_pct": 150.0}
    result = score_record(rec)
    assert result["dq_score"] < 100


def test_score_never_negative() -> None:
    rec = {"hour": 99, "month": 99, "day_of_week": 99, "consumption_kwh": -100.0, "humidity_pct": 200.0}
    result = score_record(rec)
    assert result["dq_score"] >= 0


def test_batch_score_returns_same_length() -> None:
    records = [GOOD_RECORD, {**GOOD_RECORD, "hour": 99}]
    result = batch_score(records)
    assert len(result) == 2


def test_quality_summary_mean_score() -> None:
    scored = batch_score([GOOD_RECORD, GOOD_RECORD])
    s = quality_summary(scored)
    assert s["mean_score"] == 100.0
    assert s["n_perfect"] == 2
    assert s["n_failing"] == 0


def test_quality_summary_empty() -> None:
    s = quality_summary([])
    assert s["mean_score"] == 0.0
    assert s["n_perfect"] == 0


def test_flag_outliers_detects_extreme_value() -> None:
    records = [
        {"kwh": 10.0}, {"kwh": 10.5}, {"kwh": 9.8},
        {"kwh": 10.1}, {"kwh": 10.2}, {"kwh": 500.0},
    ]
    outliers = flag_outliers(records, "kwh", z_threshold=2.0)
    assert len(outliers) == 1
    assert outliers[0]["kwh"] == 500.0


def test_flag_outliers_empty_when_constant() -> None:
    records = [{"kwh": 10.0}] * 5
    outliers = flag_outliers(records, "kwh")
    assert outliers == []


def test_flag_outliers_missing_field_skipped() -> None:
    records = [{"kwh": 10.0}, {"other": 99.0}]
    outliers = flag_outliers(records, "kwh")
    assert outliers == []


@pytest.mark.parametrize("hour", [0, 12, 23])
def test_valid_hours_score_100(hour) -> None:
    rec = {**GOOD_RECORD, "hour": hour}
    result = score_record(rec)
    assert result["dq_score"] == 100


@pytest.mark.parametrize("month", [1, 6, 12])
def test_valid_months_score_100(month: int) -> None:
    rec = {**GOOD_RECORD, "month": month}
    result = score_record(rec)
    assert result["dq_score"] == 100


@pytest.mark.parametrize("month", [0, 13, 99])
def test_invalid_months_deduct_points(month: int) -> None:
    rec = {**GOOD_RECORD, "month": month}
    result = score_record(rec)
    assert result["dq_score"] < 100


@pytest.mark.parametrize("dow", [0, 3, 6])
def test_valid_dow_score_100(dow: int) -> None:
    rec = {**GOOD_RECORD, "day_of_week": dow}
    result = score_record(rec)
    assert result["dq_score"] == 100


def test_batch_score_empty_list() -> None:
    assert batch_score([]) == []


def test_score_record_preserves_original_fields() -> None:
    result = score_record(GOOD_RECORD)
    for key in GOOD_RECORD:
        assert key in result


def test_quality_summary_has_required_keys() -> None:
    scored = batch_score([GOOD_RECORD])
    s = quality_summary(scored)
    for key in ("mean_score", "n_perfect", "n_failing"):
        assert key in s, f"Missing key: {key}"


def test_flag_outliers_high_threshold_no_outliers() -> None:
    records = [{"kwh": 10.0 + i} for i in range(5)]
    outliers = flag_outliers(records, "kwh", z_threshold=100.0)
    assert outliers == []


def test_detect_duplicates_no_duplicates() -> None:
    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert detect_duplicates(records, key_fields=["id"]) == []


def test_detect_duplicates_with_duplicates() -> None:
    records = [{"id": 1}, {"id": 2}, {"id": 1}]
    dupes = detect_duplicates(records, key_fields=["id"])
    assert 2 in dupes


def test_detect_duplicates_empty() -> None:
    assert detect_duplicates([]) == []


def test_detect_duplicates_all_fields() -> None:
    records = [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 3}]
    dupes = detect_duplicates(records)
    assert dupes == [1]


@pytest.mark.parametrize("n_dupes,expected_count", [(0, 0), (1, 1), (3, 3)])
def test_detect_duplicates_count(n_dupes: int, expected_count: int) -> None:
    records = [{"id": 0}] * (n_dupes + 1)
    assert len(detect_duplicates(records, key_fields=["id"])) == expected_count


def test_completeness_score_all_complete() -> None:
    records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert completeness_score(records, required_fields=["a", "b"]) == pytest.approx(1.0)


def test_completeness_score_none_complete() -> None:
    records = [{"a": None}, {"a": None}]
    assert completeness_score(records, required_fields=["a"]) == pytest.approx(0.0)


def test_completeness_score_partial() -> None:
    records = [{"a": 1}, {"a": None}]
    assert completeness_score(records, required_fields=["a"]) == pytest.approx(0.5)


def test_completeness_score_empty_records() -> None:
    assert completeness_score([], required_fields=["a"]) == 0.0


def test_detect_data_gaps_no_gaps() -> None:
    from app.data_quality import detect_data_gaps
    ts = [0, 3600, 7200, 10800]
    assert detect_data_gaps(ts, expected_interval=3600) == []


def test_detect_data_gaps_single_gap() -> None:
    from app.data_quality import detect_data_gaps
    ts = [0, 3600, 10800]  # gap of 2h between index 1 and 2
    gaps = detect_data_gaps(ts, expected_interval=3600)
    assert len(gaps) == 1
    assert gaps[0] == (3600, 10800)


def test_detect_data_gaps_empty() -> None:
    from app.data_quality import detect_data_gaps
    assert detect_data_gaps([]) == []


def test_detect_data_gaps_single_element() -> None:
    from app.data_quality import detect_data_gaps
    assert detect_data_gaps([1000]) == []


def test_detect_data_gaps_multiple_gaps() -> None:
    from app.data_quality import detect_data_gaps
    ts = [0, 7200, 14400, 21600]  # all gaps are 2h, expected 1h
    gaps = detect_data_gaps(ts, expected_interval=3600)
    assert len(gaps) == 3


def test_field_type_consistency_all_match() -> None:
    records = [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}]
    assert field_type_consistency(records, "value", float) == pytest.approx(1.0)


def test_field_type_consistency_none_match() -> None:
    records = [{"value": "text"}, {"value": "more"}]
    assert field_type_consistency(records, "value", float) == pytest.approx(0.0)


def test_field_type_consistency_no_field() -> None:
    records = [{"other": 1}, {"other": 2}]
    assert field_type_consistency(records, "value", float) == pytest.approx(1.0)


def test_field_type_consistency_mixed() -> None:
    records = [{"x": 1.0}, {"x": "str"}, {"x": 3.0}]
    result = field_type_consistency(records, "x", float)
    assert result == pytest.approx(2 / 3, rel=1e-4)


@pytest.mark.parametrize("expected_type,score", [
    (float, 1.0),
    (str, 0.0),
])
def test_field_type_consistency_parametrize(expected_type, score) -> None:
    records = [{"v": 1.0}, {"v": 2.0}]
    assert field_type_consistency(records, "v", expected_type) == pytest.approx(score)


def test_range_violation_count_min() -> None:
    records = [{"x": 1.0}, {"x": 5.0}, {"x": 10.0}]
    assert range_violation_count(records, "x", min_val=3.0) == 1


def test_range_violation_count_max() -> None:
    records = [{"x": 1.0}, {"x": 5.0}, {"x": 10.0}]
    assert range_violation_count(records, "x", max_val=6.0) == 1


def test_range_violation_count_none_violate() -> None:
    records = [{"x": 3.0}, {"x": 4.0}]
    assert range_violation_count(records, "x", min_val=1.0, max_val=10.0) == 0


def test_range_violation_count_missing_field() -> None:
    records = [{"y": 1.0}, {"y": 2.0}]
    assert range_violation_count(records, "x", min_val=0.0) == 0


def test_fill_missing_fills_none() -> None:
    records = [{"x": None}, {"x": 5.0}]
    result = fill_missing(records, "x", fill_value=0.0)
    assert result[0]["x"] == 0.0
    assert result[1]["x"] == 5.0


def test_fill_missing_fills_absent_key() -> None:
    records = [{"y": 1.0}]
    result = fill_missing(records, "x", fill_value=42.0)
    assert result[0]["x"] == 42.0


def test_fill_missing_does_not_modify_original() -> None:
    records = [{"x": None}]
    fill_missing(records, "x", fill_value=99.0)
    assert records[0]["x"] is None


def test_fill_missing_empty_records() -> None:
    assert fill_missing([], "x", fill_value=0.0) == []


# Tests for unique_values and records_missing_field
from app.data_quality import records_missing_field, unique_values


def test_unique_values_basic() -> None:
    records = [{"x": 1}, {"x": 2}, {"x": 1}, {"x": 3}]
    result = unique_values(records, "x")
    assert result == [1, 2, 3]


def test_unique_values_excludes_none() -> None:
    records = [{"x": 1}, {"x": None}, {"x": 2}]
    result = unique_values(records, "x")
    assert None not in result


def test_unique_values_missing_field() -> None:
    records = [{"y": 1}, {"y": 2}]
    result = unique_values(records, "x")
    assert result == []


def test_unique_values_empty_records() -> None:
    assert unique_values([], "x") == []


def test_unique_values_sorted() -> None:
    records = [{"v": 3}, {"v": 1}, {"v": 2}]
    result = unique_values(records, "v")
    assert result == [1, 2, 3]


def test_records_missing_field_all_present() -> None:
    records = [{"x": 1}, {"x": 2}, {"x": 3}]
    assert records_missing_field(records, "x") == []


def test_records_missing_field_some_missing() -> None:
    records = [{"x": 1}, {"x": None}, {"x": 3}, {}]
    result = records_missing_field(records, "x")
    assert 1 in result
    assert 3 in result


def test_records_missing_field_all_missing() -> None:
    records = [{}, {}, {}]
    result = records_missing_field(records, "x")
    assert result == [0, 1, 2]


def test_records_missing_field_empty() -> None:
    assert records_missing_field([], "x") == []


@pytest.mark.parametrize("n", [1, 3, 5])
def test_records_missing_field_count(n: int) -> None:
    records = [{"x": None}] * n
    result = records_missing_field(records, "x")
    assert len(result) == n
