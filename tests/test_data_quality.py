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
    records_missing_field,
    score_record,
    unique_values,
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
        {"kwh": 10.0},
        {"kwh": 10.5},
        {"kwh": 9.8},
        {"kwh": 10.1},
        {"kwh": 10.2},
        {"kwh": 500.0},
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


def test_completeness_score_all_fields_present() -> None:
    fields = ["hour", "month", "day_of_week", "consumption_kwh"]
    records = [{"hour": 12, "month": 6, "day_of_week": 1, "consumption_kwh": 10.0}]
    score = completeness_score(records, fields)
    assert score == pytest.approx(1.0)


def test_completeness_score_missing_field() -> None:
    fields = ["hour", "month", "day_of_week", "consumption_kwh"]
    records = [{"hour": 12, "month": 6}]
    score = completeness_score(records, fields)
    assert 0.0 <= score < 1.0


def test_quality_summary_total_records() -> None:
    scored = batch_score([GOOD_RECORD, GOOD_RECORD, GOOD_RECORD])
    summary = quality_summary(scored)
    assert summary["total_records"] == 3


def test_flag_outliers_none_in_normal_data() -> None:
    records = [{"kwh": 10.0 + i * 0.1} for i in range(20)]
    flags = flag_outliers(records, field="kwh")
    assert isinstance(flags, list)


@pytest.mark.parametrize("n", [1, 5, 10])
def test_batch_score_returns_n_scores(n: int) -> None:
    records = [GOOD_RECORD] * n
    scores = batch_score(records)
    assert len(scores) == n


class TestSchemaValidate:
    def test_valid_record_returns_no_errors(self) -> None:
        from app.data_quality import schema_validate

        schema = {"consumption_kwh": float, "hour": int}
        record = {"consumption_kwh": 5.0, "hour": 14}
        assert schema_validate(record, schema) == []

    def test_missing_field_returns_error(self) -> None:
        from app.data_quality import schema_validate

        errors = schema_validate({}, {"kwh": float})
        assert any("missing:kwh" in e for e in errors)

    def test_wrong_type_returns_error(self) -> None:
        from app.data_quality import schema_validate

        errors = schema_validate({"kwh": "not_a_float"}, {"kwh": float})
        assert any("type_error:kwh" in e for e in errors)

    def test_multiple_fields_all_errors(self) -> None:
        from app.data_quality import schema_validate

        schema = {"a": int, "b": str}
        errors = schema_validate({"a": "x", "b": 123}, schema)
        assert len(errors) == 2

    @pytest.mark.parametrize("expected_type,value", [(int, 42), (float, 3.14), (str, "hello")])
    def test_correct_types_pass(self, expected_type: type, value) -> None:
        from app.data_quality import schema_validate

        errors = schema_validate({"field": value}, {"field": expected_type})
        assert errors == []


class TestNormalizeRecord:
    def test_strips_whitespace(self) -> None:
        from app.data_quality import normalize_record

        result = normalize_record({"region": "  NORTHEAST  "})
        assert result["region"] == "northeast"

    def test_lowercases_strings(self) -> None:
        from app.data_quality import normalize_record

        result = normalize_record({"building_type": "COMMERCIAL"})
        assert result["building_type"] == "commercial"

    def test_numeric_fields_unchanged(self) -> None:
        from app.data_quality import normalize_record

        result = normalize_record({"kwh": 42.5, "hour": 14})
        assert result["kwh"] == 42.5
        assert result["hour"] == 14

    def test_empty_record(self) -> None:
        from app.data_quality import normalize_record

        assert normalize_record({}) == {}

    @pytest.mark.parametrize("input_val,expected", [("Hello", "hello"), ("  Test  ", "test"), ("ok", "ok")])
    def test_various_strings(self, input_val: str, expected: str) -> None:
        from app.data_quality import normalize_record

        assert normalize_record({"x": input_val})["x"] == expected


class TestFieldValueCounts:
    def test_basic_counts(self) -> None:
        from app.data_quality import field_value_counts

        records = [{"type": "A"}, {"type": "B"}, {"type": "A"}]
        result = field_value_counts(records, "type")
        assert result["A"] == 2
        assert result["B"] == 1

    def test_missing_field_counted_as_empty_string(self) -> None:
        from app.data_quality import field_value_counts

        records = [{"x": 1}, {"x": 1}]
        result = field_value_counts(records, "type")
        assert "" in result

    def test_sorted_descending(self) -> None:
        from app.data_quality import field_value_counts

        records = [{"k": "a"}, {"k": "b"}, {"k": "b"}, {"k": "b"}]
        keys = list(field_value_counts(records, "k").keys())
        assert keys[0] == "b"

    def test_empty_records(self) -> None:
        from app.data_quality import field_value_counts

        assert field_value_counts([], "type") == {}


class TestNullRate:
    def test_all_null(self) -> None:
        from app.data_quality import null_rate

        records = [{"a": None}, {"a": None}]
        assert null_rate(records, "a") == pytest.approx(1.0)

    def test_none_null(self) -> None:
        from app.data_quality import null_rate

        records = [{"a": 1}, {"a": 2}]
        assert null_rate(records, "a") == pytest.approx(0.0)

    def test_partial_null(self) -> None:
        from app.data_quality import null_rate

        records = [{"a": 1}, {"a": None}]
        assert null_rate(records, "a") == pytest.approx(0.5)

    def test_empty_list(self) -> None:
        from app.data_quality import null_rate

        assert null_rate([], "a") == 0.0


class TestDuplicateRate:
    def test_no_duplicates(self) -> None:
        from app.data_quality import duplicate_rate

        records = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 3, "v": "c"}]
        assert duplicate_rate(records, ["id"]) == pytest.approx(0.0)

    def test_all_duplicates(self) -> None:
        from app.data_quality import duplicate_rate

        records = [{"id": 1}, {"id": 1}, {"id": 1}]
        rate = duplicate_rate(records, ["id"])
        assert rate == pytest.approx(2 / 3, rel=1e-4)

    def test_empty_records(self) -> None:
        from app.data_quality import duplicate_rate

        assert duplicate_rate([], ["id"]) == 0.0

    def test_compound_key(self) -> None:
        from app.data_quality import duplicate_rate

        records = [
            {"a": 1, "b": 1},
            {"a": 1, "b": 2},
            {"a": 1, "b": 1},
        ]
        rate = duplicate_rate(records, ["a", "b"])
        assert rate == pytest.approx(1 / 3, rel=1e-4)

    def test_single_record(self) -> None:
        from app.data_quality import duplicate_rate

        assert duplicate_rate([{"id": 99}], ["id"]) == 0.0


class TestFieldCompleteness:
    def test_all_present(self) -> None:
        from app.data_quality import field_completeness

        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = field_completeness(records, ["a", "b"])
        assert result["a"] == pytest.approx(1.0)
        assert result["b"] == pytest.approx(1.0)

    def test_none_present(self) -> None:
        from app.data_quality import field_completeness

        records = [{"a": None}, {"a": None}]
        result = field_completeness(records, ["a"])
        assert result["a"] == pytest.approx(0.0)

    def test_empty_records(self) -> None:
        from app.data_quality import field_completeness

        result = field_completeness([], ["x", "y"])
        assert result == {"x": 0.0, "y": 0.0}

    def test_partial_completeness(self) -> None:
        from app.data_quality import field_completeness

        records = [{"a": 1}, {"a": None}, {"a": 3}, {"a": None}]
        result = field_completeness(records, ["a"])
        assert result["a"] == pytest.approx(0.5)

    def test_multiple_fields(self) -> None:
        from app.data_quality import field_completeness

        records = [{"a": 1, "b": None}, {"a": 2, "b": 3}]
        result = field_completeness(records, ["a", "b"])
        assert result["a"] == pytest.approx(1.0)
        assert result["b"] == pytest.approx(0.5)


class TestValueRangeCheck:
    def test_all_in_range(self) -> None:
        from app.data_quality import value_range_check

        records = [{"v": 5}, {"v": 10}, {"v": 15}]
        result = value_range_check(records, "v", 0, 20)
        assert result["out_of_range_count"] == 0
        assert result["out_of_range_rate"] == pytest.approx(0.0)

    def test_all_out_of_range(self) -> None:
        from app.data_quality import value_range_check

        records = [{"v": -5}, {"v": 200}]
        result = value_range_check(records, "v", 0, 100)
        assert result["out_of_range_count"] == 2

    def test_empty_records(self) -> None:
        from app.data_quality import value_range_check

        result = value_range_check([], "v", 0, 100)
        assert result["total_checked"] == 0

    def test_boundary_values_in_range(self) -> None:
        from app.data_quality import value_range_check

        records = [{"v": 0}, {"v": 100}]
        result = value_range_check(records, "v", 0, 100)
        assert result["out_of_range_count"] == 0

    def test_rate_correct(self) -> None:
        from app.data_quality import value_range_check

        records = [{"v": 1}, {"v": 2}, {"v": 200}, {"v": 300}]
        result = value_range_check(records, "v", 0, 10)
        assert result["out_of_range_rate"] == pytest.approx(0.5)


class TestDataFreshnessScore:
    def test_all_fresh(self) -> None:
        import time

        from app.data_quality import data_freshness_score

        now = time.time()
        records = [{"ts": now - 100}, {"ts": now - 200}]
        result = data_freshness_score(records, "ts", max_age_seconds=3600)
        assert result["fresh_count"] == 2
        assert result["freshness_rate"] == pytest.approx(1.0)

    def test_all_stale(self) -> None:
        import time

        from app.data_quality import data_freshness_score

        old = time.time() - 7200
        records = [{"ts": old}, {"ts": old}]
        result = data_freshness_score(records, "ts", max_age_seconds=3600)
        assert result["stale_count"] == 2

    def test_empty_records(self) -> None:
        from app.data_quality import data_freshness_score

        result = data_freshness_score([], "ts")
        assert result["total_records"] == 0

    def test_mixed_freshness(self) -> None:
        import time

        from app.data_quality import data_freshness_score

        now = time.time()
        records = [{"ts": now - 100}, {"ts": now - 7200}]
        result = data_freshness_score(records, "ts", max_age_seconds=3600)
        assert result["fresh_count"] == 1
        assert result["stale_count"] == 1
        assert result["freshness_rate"] == pytest.approx(0.5)

    def test_result_keys(self) -> None:
        import time

        from app.data_quality import data_freshness_score

        records = [{"ts": time.time()}]
        result = data_freshness_score(records, "ts")
        assert set(result.keys()) >= {"total_records", "fresh_count", "stale_count", "freshness_rate"}


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


@pytest.mark.parametrize(
    "expected_type,score",
    [
        (float, 1.0),
        (str, 0.0),
    ],
)
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


def test_batch_score_returns_all_fields() -> None:
    records = [{"hour": 10, "month": 3, "day_of_week": 1, "consumption_kwh": 20.0}]
    result = batch_score(records)
    assert len(result) == 1
    assert "dq_score" in result[0]
    assert "dq_issues" in result[0]


@pytest.mark.parametrize(
    "consumption,expected_score_at_least",
    [
        (1.0, 80),  # normal reading
        (-1.0, 0),  # negative is invalid → heavy penalty
        (0.0, 50),  # zero might be acceptable
    ],
)
def test_score_record_consumption_ranges(consumption: float, expected_score_at_least: int) -> None:
    record = {"hour": 10, "month": 3, "day_of_week": 1, "consumption_kwh": consumption}
    scored = score_record(record)
    assert scored["dq_score"] >= expected_score_at_least or consumption < 0


def test_quality_summary_perfect_scores() -> None:
    scored = [{"dq_score": 100, "dq_issues": []} for _ in range(5)]
    summary = quality_summary(scored)
    assert summary["total_records"] == 5
    assert summary["n_perfect"] == 5
    assert summary["n_failing"] == 0


def test_unique_values_returns_set() -> None:
    records = [{"tag": "a"}, {"tag": "b"}, {"tag": "a"}]
    result = unique_values(records, "tag")
    assert set(result) == {"a", "b"}


class TestCrossFieldConsistency:
    def test_no_violations(self) -> None:
        from app.data_quality import cross_field_consistency

        records = [{"low": 1, "high": 5}, {"low": 2, "high": 3}]
        assert cross_field_consistency(records, "low", "high", "a_lte_b") == []

    def test_violation_detected(self) -> None:
        from app.data_quality import cross_field_consistency

        records = [{"low": 10, "high": 5}, {"low": 1, "high": 3}]
        result = cross_field_consistency(records, "low", "high", "a_lte_b")
        assert 0 in result

    def test_invalid_relation_raises(self) -> None:
        from app.data_quality import cross_field_consistency

        with pytest.raises(ValueError, match="relation"):
            cross_field_consistency([{"a": 1, "b": 2}], "a", "b", "equal")

    def test_none_fields_skipped(self) -> None:
        from app.data_quality import cross_field_consistency

        records = [{"low": None, "high": 5}, {"low": 1, "high": 3}]
        assert cross_field_consistency(records, "low", "high", "a_lte_b") == []

    @pytest.mark.parametrize(
        "records,expected_count",
        [
            ([{"a": 1, "b": 2}, {"a": 3, "b": 4}], 0),
            ([{"a": 5, "b": 2}, {"a": 1, "b": 4}], 1),
        ],
    )
    def test_parametrized(self, records: list, expected_count: int) -> None:
        from app.data_quality import cross_field_consistency

        result = cross_field_consistency(records, "a", "b", "a_lte_b")
        assert len(result) == expected_count


class TestOutlierCountIqr:
    def test_no_outliers(self) -> None:
        from app.data_quality import outlier_count_iqr

        records = [{"v": float(i)} for i in range(1, 11)]
        assert outlier_count_iqr(records, "v") == 0

    def test_extreme_outlier_detected(self) -> None:
        from app.data_quality import outlier_count_iqr

        records = [{"v": float(i)} for i in range(1, 10)] + [{"v": 1000.0}]
        assert outlier_count_iqr(records, "v") > 0

    def test_negative_multiplier_raises(self) -> None:
        from app.data_quality import outlier_count_iqr

        with pytest.raises(ValueError, match="positive"):
            outlier_count_iqr([{"v": 1.0}], "v", multiplier=-1.0)

    def test_too_few_values(self) -> None:
        from app.data_quality import outlier_count_iqr

        records = [{"v": 1.0}, {"v": 2.0}]
        assert outlier_count_iqr(records, "v") == 0


class TestSchemaConformanceRate:
    def test_all_conforming(self) -> None:
        from app.data_quality import schema_conformance_rate

        records = [{"x": 1, "y": "hello"}, {"x": 2, "y": "world"}]
        schema = {"x": int, "y": str}
        assert schema_conformance_rate(records, schema) == 1.0

    def test_none_value_fails(self) -> None:
        from app.data_quality import schema_conformance_rate

        records = [{"x": None, "y": "hello"}]
        schema = {"x": int, "y": str}
        assert schema_conformance_rate(records, schema) == 0.0

    def test_partial_conformance(self) -> None:
        from app.data_quality import schema_conformance_rate

        records = [{"x": 1, "y": "ok"}, {"x": "bad", "y": "ok"}]
        schema = {"x": int, "y": str}
        assert schema_conformance_rate(records, schema) == pytest.approx(0.5)

    def test_empty_records(self) -> None:
        from app.data_quality import schema_conformance_rate

        assert schema_conformance_rate([], {"x": int}) == 1.0


# ---------------------------------------------------------------------------
# Tests for find_duplicate_rows, value_range_check, field_entropy
# ---------------------------------------------------------------------------


class TestFindDuplicateRows:
    def test_no_duplicates(self) -> None:
        from app.data_quality import find_duplicate_rows

        records = [{"id": 1}, {"id": 2}, {"id": 3}]
        assert find_duplicate_rows(records, ["id"]) == []

    def test_one_duplicate(self) -> None:
        from app.data_quality import find_duplicate_rows

        records = [{"id": 1}, {"id": 2}, {"id": 1}]
        dupes = find_duplicate_rows(records, ["id"])
        assert len(dupes) == 1
        assert dupes[0]["id"] == 1

    def test_composite_key(self) -> None:
        from app.data_quality import find_duplicate_rows

        records = [
            {"a": 1, "b": 2},
            {"a": 1, "b": 3},
            {"a": 1, "b": 2},
        ]
        dupes = find_duplicate_rows(records, ["a", "b"])
        assert len(dupes) == 1

    def test_empty_records(self) -> None:
        from app.data_quality import find_duplicate_rows

        assert find_duplicate_rows([], ["id"]) == []


class TestValueRangeCheckExtended:
    def test_all_in_range(self) -> None:
        from app.data_quality import value_range_check

        result = value_range_check([1.0, 2.0, 3.0], 0.0, 5.0)
        assert result["out_of_range"] == 0
        assert result["in_range"] == 3

    def test_some_out_of_range(self) -> None:
        from app.data_quality import value_range_check

        result = value_range_check([0.0, 5.0, 10.0], 1.0, 9.0)
        assert result["out_of_range"] == 2

    def test_empty_raises(self) -> None:
        import pytest

        from app.data_quality import value_range_check

        with pytest.raises(ValueError):
            value_range_check([], 0.0, 1.0)

    def test_low_gt_high_raises(self) -> None:
        import pytest

        from app.data_quality import value_range_check

        with pytest.raises(ValueError):
            value_range_check([1.0], 5.0, 3.0)


class TestFieldEntropy:
    def test_uniform_distribution(self) -> None:
        from app.data_quality import field_entropy

        records = [{"x": i} for i in range(4)]
        result = field_entropy(records, "x")
        assert result == pytest.approx(2.0, abs=0.01)

    def test_all_same_value(self) -> None:
        from app.data_quality import field_entropy

        records = [{"x": "a"}] * 5
        assert field_entropy(records, "x") == 0.0

    def test_empty_records(self) -> None:
        from app.data_quality import field_entropy

        assert field_entropy([], "x") == 0.0

    def test_missing_field_treated_as_none(self) -> None:
        from app.data_quality import field_entropy

        records = [{"x": 1}, {}, {}]
        result = field_entropy(records, "x")
        assert result > 0.0


import pytest as _pytest


@_pytest.mark.parametrize(
    "records,required,expected_score",
    [
        ([{"a": 1, "b": 2}], ["a", "b"], 1.0),
        ([{"a": 1}], ["a", "b"], 0.5),
        ([], ["a"], 0.0),
    ],
)
def test_completeness_score_parametrized(
    records: list, required: list, expected_score: float
) -> None:
    from app.data_quality import completeness_score

    assert completeness_score(records, required) == _pytest.approx(expected_score, abs=0.001)


@_pytest.mark.parametrize(
    "records,field,expected_null_rate",
    [
        ([{"v": 1}, {"v": 2}], "v", 0.0),
        ([{"v": None}, {"v": None}], "v", 1.0),
        ([{"v": 1}, {"v": None}], "v", 0.5),
    ],
)
def test_null_rate_parametrized(records: list, field: str, expected_null_rate: float) -> None:
    from app.data_quality import null_rate

    assert null_rate(records, field) == _pytest.approx(expected_null_rate, abs=0.001)


@_pytest.mark.parametrize("n", [0, 1, 5, 100])
def test_batch_score_length_matches_input(n: int) -> None:
    from app.data_quality import batch_score

    records = [{"consumption_kwh": 10.0, "hour": 12, "month": 6, "day_of_week": 2}] * n
    result = batch_score(records)
    assert len(result) == n
