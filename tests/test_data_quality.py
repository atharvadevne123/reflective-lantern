"""Tests for app/data_quality.py."""

from __future__ import annotations

import pytest

from app.data_quality import batch_score, flag_outliers, quality_summary, score_record

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
