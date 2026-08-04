"""Tests for app/data_quality.py."""

from __future__ import annotations

import pytest

from app.data_quality import (
    batch_score,
    completeness_score,
    detect_duplicates,
    flag_outliers,
    quality_summary,
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
