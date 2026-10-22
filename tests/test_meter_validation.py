"""Tests for app.meter_validation module."""

from __future__ import annotations

import pytest

from app.meter_validation import (
    ValidationReport,
    ValidationStatus,
    check_negatives,
    check_upper_bound,
    check_zeros,
    detect_gaps,
    validate_readings,
)


class TestCheckNegatives:
    def test_no_negatives(self) -> None:
        assert check_negatives([1.0, 2.0, 3.0]) == []

    def test_single_negative(self) -> None:
        issues = check_negatives([1.0, -0.5, 2.0])
        assert len(issues) == 1
        assert issues[0].index == 1
        assert issues[0].kind == "negative_reading"

    def test_all_negative(self) -> None:
        issues = check_negatives([-1.0, -2.0])
        assert len(issues) == 2

    def test_zero_is_not_negative(self) -> None:
        assert check_negatives([0.0, 1.0]) == []

    @pytest.mark.parametrize(
        "readings,expected_count",
        [([0.5, 1.0, -0.1], 1), ([-1.0, -2.0, -3.0], 3), ([1.0, 2.0], 0)],
    )
    def test_parametrized(self, readings: list[float], expected_count: int) -> None:
        assert len(check_negatives(readings)) == expected_count


class TestCheckUpperBound:
    def test_no_violations(self) -> None:
        assert check_upper_bound([10.0, 20.0], upper=500.0) == []

    def test_single_violation(self) -> None:
        issues = check_upper_bound([1.0, 600.0, 2.0], upper=500.0)
        assert len(issues) == 1
        assert issues[0].index == 1

    def test_custom_upper(self) -> None:
        issues = check_upper_bound([5.0, 15.0], upper=10.0)
        assert len(issues) == 1

    @pytest.mark.parametrize(
        "readings,upper,expected",
        [([5.0, 4.0, 3.0], 10.0, 0), ([100.0, 200.0], 150.0, 1)],
    )
    def test_parametrized(self, readings: list[float], upper: float, expected: int) -> None:
        assert len(check_upper_bound(readings, upper)) == expected


class TestCheckZeros:
    def test_no_zeros_no_issues(self) -> None:
        assert check_zeros([1.0, 2.0, 3.0]) == []

    def test_short_zero_run_ok(self) -> None:
        assert check_zeros([0.0, 0.0, 0.0, 0.0, 1.0], max_consecutive_zeros=4) == []

    def test_long_zero_run_flagged(self) -> None:
        issues = check_zeros([0.0] * 6 + [1.0], max_consecutive_zeros=4)
        assert len(issues) == 6

    def test_zero_run_at_end(self) -> None:
        issues = check_zeros([1.0] + [0.0] * 5, max_consecutive_zeros=4)
        assert len(issues) == 5

    @pytest.mark.parametrize(
        "readings,max_z,expected_issues",
        [
            ([0.0, 0.0, 0.0, 0.0], 3, 4),
            ([0.0, 0.0, 1.0, 0.0], 3, 0),
        ],
    )
    def test_parametrized(self, readings: list[float], max_z: int, expected_issues: int) -> None:
        assert len(check_zeros(readings, max_z)) == expected_issues


class TestDetectGaps:
    def test_no_gaps(self) -> None:
        ts = [0, 1800, 3600, 5400]
        assert detect_gaps(ts, 1800) == []

    def test_single_gap(self) -> None:
        ts = [0, 1800, 10000, 11800]
        gaps = detect_gaps(ts, 1800)
        assert 2 in gaps

    def test_fewer_than_two_returns_empty(self) -> None:
        assert detect_gaps([12345]) == []

    def test_unsorted_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_gaps([3600, 1800, 5400])

    @pytest.mark.parametrize(
        "ts,interval,expected_gaps",
        [
            ([0, 1800, 3600], 1800, []),
            ([0, 5000], 1800, [1]),
        ],
    )
    def test_parametrized(self, ts: list[int], interval: int, expected_gaps: list[int]) -> None:
        assert detect_gaps(ts, interval) == expected_gaps


class TestValidateReadings:
    def test_valid_readings(self) -> None:
        result = validate_readings([1.0, 2.0, 3.0])
        assert result.status == ValidationStatus.VALID
        assert result.issues == []

    def test_empty_readings_invalid(self) -> None:
        result = validate_readings([])
        assert result.status == ValidationStatus.INVALID

    def test_negative_marks_invalid(self) -> None:
        result = validate_readings([1.0, -0.5, 2.0])
        assert result.status == ValidationStatus.INVALID

    def test_upper_bound_violation_warning(self) -> None:
        result = validate_readings([1.0, 1000.0, 2.0])
        assert result.status in (ValidationStatus.WARNINGS, ValidationStatus.INVALID)

    def test_returns_validation_report(self) -> None:
        result = validate_readings([1.0, 2.0])
        assert isinstance(result, ValidationReport)

    def test_total_readings_count(self) -> None:
        readings = [1.0] * 10
        result = validate_readings(readings)
        assert result.total_readings == 10

    @pytest.mark.parametrize(
        "readings,expected_status",
        [
            ([1.0, 2.0, 3.0], ValidationStatus.VALID),
            ([-1.0, 2.0], ValidationStatus.INVALID),
        ],
    )
    def test_parametrized_status(
        self, readings: list[float], expected_status: ValidationStatus
    ) -> None:
        result = validate_readings(readings)
        assert result.status == expected_status
