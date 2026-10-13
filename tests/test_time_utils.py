"""Tests for app.time_utils module."""

from __future__ import annotations

import datetime

import pytest

from app.time_utils import (
    format_duration_seconds,
    is_business_hours,
    iso_now,
    parse_iso,
    seconds_until,
    utc_now,
)


class TestUtcNow:
    def test_returns_datetime(self) -> None:
        result = utc_now()
        assert isinstance(result, datetime.datetime)

    def test_is_timezone_aware(self) -> None:
        result = utc_now()
        assert result.tzinfo is not None

    def test_utc_timezone(self) -> None:
        result = utc_now()
        assert result.utcoffset() == datetime.timedelta(0)


class TestIsoNow:
    def test_returns_string(self) -> None:
        result = iso_now()
        assert isinstance(result, str)

    def test_parseable(self) -> None:
        result = iso_now()
        dt = datetime.datetime.fromisoformat(result)
        assert dt is not None

    def test_contains_date_separator(self) -> None:
        result = iso_now()
        assert "T" in result or "-" in result


class TestParseIso:
    def test_parse_utc_string(self) -> None:
        result = parse_iso("2024-01-15T10:30:00+00:00")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_z_suffix(self) -> None:
        result = parse_iso("2024-06-01T12:00:00Z")
        assert result.hour == 12

    def test_naive_gets_utc(self) -> None:
        result = parse_iso("2024-03-01T09:00:00")
        assert result.tzinfo == datetime.UTC

    @pytest.mark.parametrize(
        "ts",
        [
            "2024-01-01T00:00:00+00:00",
            "2024-12-31T23:59:59Z",
            "2024-07-04T18:00:00",
        ],
    )
    def test_various_formats(self, ts: str) -> None:
        result = parse_iso(ts)
        assert isinstance(result, datetime.datetime)


class TestSecondsUntil:
    def test_future_is_positive(self) -> None:
        future = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=1)
        assert seconds_until(future) > 0

    def test_past_is_negative(self) -> None:
        past = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=1)
        assert seconds_until(past) < 0

    def test_approximate_value(self) -> None:
        future = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(seconds=60)
        remaining = seconds_until(future)
        assert 55 < remaining <= 60


class TestFormatDurationSeconds:
    def test_seconds_only(self) -> None:
        assert format_duration_seconds(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert format_duration_seconds(90) == "1m 30s"

    def test_hours_minutes_seconds(self) -> None:
        assert format_duration_seconds(3661) == "1h 1m 1s"

    def test_zero_seconds(self) -> None:
        assert format_duration_seconds(0) == "0s"

    def test_exact_hour(self) -> None:
        result = format_duration_seconds(3600)
        assert "1h" in result

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (60, "1m 0s"),
            (3600, "1h 0s"),
            (7200, "2h 0s"),
        ],
    )
    def test_parametrized(self, seconds: float, expected: str) -> None:
        assert format_duration_seconds(seconds) == expected


class TestIsBusinessHours:
    def test_weekday_in_hours(self) -> None:
        monday_10am = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=datetime.UTC)
        assert is_business_hours(monday_10am) is True

    def test_weekday_before_hours(self) -> None:
        monday_8am = datetime.datetime(2024, 1, 15, 7, 0, tzinfo=datetime.UTC)
        assert is_business_hours(monday_8am) is False

    def test_weekday_after_hours(self) -> None:
        monday_6pm = datetime.datetime(2024, 1, 15, 18, 0, tzinfo=datetime.UTC)
        assert is_business_hours(monday_6pm) is False

    def test_saturday_returns_false(self) -> None:
        saturday = datetime.datetime(2024, 1, 20, 12, 0, tzinfo=datetime.UTC)
        assert is_business_hours(saturday) is False

    def test_sunday_returns_false(self) -> None:
        sunday = datetime.datetime(2024, 1, 21, 12, 0, tzinfo=datetime.UTC)
        assert is_business_hours(sunday) is False

    @pytest.mark.parametrize("hour", [9, 12, 16])
    def test_valid_business_hours(self, hour: int) -> None:
        monday = datetime.datetime(2024, 1, 15, hour, 0, tzinfo=datetime.UTC)
        assert is_business_hours(monday) is True
