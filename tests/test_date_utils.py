"""Tests for app/date_utils.py."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.date_utils import clamp_to_range, start_of_month


def test_utc_now_is_aware() -> None:
    from app.date_utils import utc_now

    now = utc_now()
    assert now.tzinfo is not None


def test_round_to_hour_zeroes_minutes() -> None:
    from app.date_utils import round_to_hour

    dt = datetime(2026, 6, 15, 14, 47, 33)
    result = round_to_hour(dt)
    assert result.minute == 0
    assert result.second == 0
    assert result.hour == 14


def test_hours_between_same() -> None:
    from app.date_utils import hours_between

    dt = datetime(2026, 1, 1, 12)
    assert hours_between(dt, dt) == 0


def test_hours_between_24h() -> None:
    from datetime import timedelta

    from app.date_utils import hours_between

    start = datetime(2026, 1, 1, 0)
    end = start + timedelta(hours=24)
    assert hours_between(start, end) == 24


def test_iso_to_datetime_basic() -> None:
    from app.date_utils import iso_to_datetime

    result = iso_to_datetime("2026-06-15T14:30:00")
    assert result.year == 2026
    assert result.hour == 14


def test_iso_to_datetime_with_z() -> None:
    from app.date_utils import iso_to_datetime

    result = iso_to_datetime("2026-06-15T14:30:00Z")
    assert result.tzinfo is not None


def test_iso_to_datetime_invalid_raises() -> None:
    from app.date_utils import iso_to_datetime

    with pytest.raises(ValueError, match="Cannot parse"):
        iso_to_datetime("not-a-date")


def test_generate_hourly_timestamps_length() -> None:
    from app.date_utils import generate_hourly_timestamps

    start = datetime(2026, 1, 1)
    result = generate_hourly_timestamps(start, 24)
    assert len(result) == 24


def test_generate_hourly_timestamps_spacing() -> None:
    from datetime import timedelta

    from app.date_utils import generate_hourly_timestamps

    start = datetime(2026, 1, 1)
    result = generate_hourly_timestamps(start, 3)
    assert result[1] - result[0] == timedelta(hours=1)


def test_generate_hourly_timestamps_zero_raises() -> None:
    from app.date_utils import generate_hourly_timestamps

    with pytest.raises(ValueError):
        generate_hourly_timestamps(datetime(2026, 1, 1), 0)


def test_is_business_hour_true() -> None:
    from app.date_utils import is_business_hour

    dt = datetime(2026, 6, 15, 10, 0)  # Monday 10am
    assert is_business_hour(dt) is True


def test_is_business_hour_false_weekend() -> None:
    from app.date_utils import is_business_hour

    dt = datetime(2026, 6, 14, 10, 0)  # Sunday
    assert is_business_hour(dt) is False


def test_is_business_hour_false_early() -> None:
    from app.date_utils import is_business_hour

    dt = datetime(2026, 6, 15, 7, 0)  # Monday 7am
    assert is_business_hour(dt) is False


def test_week_of_year() -> None:
    from app.date_utils import week_of_year

    dt = datetime(2026, 1, 5)  # First week
    assert 1 <= week_of_year(dt) <= 2


@pytest.mark.parametrize("n_hours", [1, 6, 24, 168])
def test_generate_hourly_timestamps_various_lengths(n_hours) -> None:
    from app.date_utils import generate_hourly_timestamps

    result = generate_hourly_timestamps(datetime(2026, 1, 1), n_hours)
    assert len(result) == n_hours


@pytest.mark.parametrize("n_hours", [1, 6, 24, 168])
def test_generate_hourly_timestamps_parametrized(n_hours: int) -> None:
    from datetime import datetime

    from app.date_utils import generate_hourly_timestamps

    start = datetime(2026, 1, 1, 0)
    result = generate_hourly_timestamps(start, n_hours)
    assert len(result) == n_hours


def test_generate_hourly_timestamps_hourly_gaps() -> None:
    from datetime import datetime, timedelta

    from app.date_utils import generate_hourly_timestamps

    start = datetime(2026, 6, 1, 12)
    result = generate_hourly_timestamps(start, 5)
    for i in range(1, len(result)):
        diff = result[i] - result[i - 1]
        assert diff == timedelta(hours=1)


def test_generate_hourly_timestamps_invalid_n() -> None:
    from datetime import datetime

    from app.date_utils import generate_hourly_timestamps

    with pytest.raises(ValueError):
        generate_hourly_timestamps(datetime(2026, 1, 1), 0)


@pytest.mark.parametrize(
    "hour,expected",
    [
        (8, True),
        (12, True),
        (17, True),
        (7, False),
        (18, False),
        (22, False),
    ],
)
def test_is_business_hour_parametrized(hour: int, expected: bool) -> None:
    from datetime import datetime

    from app.date_utils import is_business_hour

    dt = datetime(2026, 7, 20, hour)  # Monday
    assert is_business_hour(dt) == expected


def test_is_business_hour_weekend_false() -> None:
    from datetime import datetime

    from app.date_utils import is_business_hour

    saturday = datetime(2026, 7, 18, 12)  # Saturday noon
    assert not is_business_hour(saturday)


def test_week_of_year_january_first() -> None:
    from datetime import datetime

    from app.date_utils import week_of_year

    dt = datetime(2026, 1, 5)  # First week of 2026
    assert 1 <= week_of_year(dt) <= 2


@pytest.mark.parametrize(
    "month,day,expected_min,expected_max",
    [
        (1, 1, 1, 2),
        (6, 15, 24, 26),
        (12, 31, 52, 54),
    ],
)
def test_week_of_year_parametrized(month, day, expected_min, expected_max) -> None:
    from datetime import datetime

    from app.date_utils import week_of_year

    dt = datetime(2026, month, day)
    assert expected_min <= week_of_year(dt) <= expected_max


def test_hours_between_negative_is_negative() -> None:
    from datetime import datetime

    from app.date_utils import hours_between

    start = datetime(2026, 1, 2, 0)
    end = datetime(2026, 1, 1, 0)
    result = hours_between(start, end)
    assert result <= 0


def test_round_to_hour_already_on_hour() -> None:
    from datetime import datetime

    from app.date_utils import round_to_hour

    dt = datetime(2026, 6, 1, 9, 0, 0)
    result = round_to_hour(dt)
    assert result.hour == 9
    assert result.minute == 0


@pytest.mark.parametrize("h", [0, 6, 12, 18, 23])
def test_is_business_hour_boundary_hours(h: int) -> None:
    from datetime import datetime

    from app.date_utils import is_business_hour

    dt = datetime(2026, 7, 20, h)  # Monday
    result = is_business_hour(dt)
    assert isinstance(result, bool)


def test_generate_hourly_timestamps_first_element_matches_start() -> None:
    from datetime import datetime

    from app.date_utils import generate_hourly_timestamps

    start = datetime(2026, 3, 15, 8)
    result = generate_hourly_timestamps(start, 10)
    assert result[0] == start


class TestDatetimeToIso:
    def test_naive_becomes_utc(self) -> None:
        from datetime import datetime

        from app.date_utils import datetime_to_iso

        dt = datetime(2026, 8, 3, 12, 0, 0)
        result = datetime_to_iso(dt)
        assert "+00:00" in result

    def test_aware_preserved(self) -> None:
        from datetime import datetime, timedelta, timezone

        from app.date_utils import datetime_to_iso

        tz = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2026, 8, 3, 17, 30, 0, tzinfo=tz)
        assert "+05:30" in datetime_to_iso(dt)

    def test_returns_string(self) -> None:
        from datetime import datetime

        from app.date_utils import datetime_to_iso

        assert isinstance(datetime_to_iso(datetime(2026, 1, 1)), str)


class TestStartOfDay:
    def test_zeroes_time(self) -> None:
        from datetime import datetime

        from app.date_utils import start_of_day

        dt = datetime(2026, 8, 3, 15, 45, 30)
        result = start_of_day(dt)
        assert result.hour == 0 and result.minute == 0 and result.second == 0

    def test_preserves_date(self) -> None:
        from datetime import datetime

        from app.date_utils import start_of_day

        dt = datetime(2026, 8, 3, 15, 45)
        result = start_of_day(dt)
        assert result.date() == dt.date()


class TestDaysBetween:
    def test_positive(self) -> None:
        from datetime import datetime

        from app.date_utils import days_between

        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 8)
        assert days_between(start, end) == 7

    def test_same_day(self) -> None:
        from datetime import datetime

        from app.date_utils import days_between

        dt = datetime(2026, 6, 1)
        assert days_between(dt, dt) == 0

    def test_negative(self) -> None:
        from datetime import datetime

        from app.date_utils import days_between

        start = datetime(2026, 2, 1)
        end = datetime(2026, 1, 1)
        assert days_between(start, end) == -31


class TestFormatDuration:
    def test_seconds_only(self) -> None:
        from app.date_utils import format_duration

        assert format_duration(45.0) == "45s"

    def test_minutes_and_seconds(self) -> None:
        from app.date_utils import format_duration

        assert format_duration(125.0) == "2m 5s"

    def test_hours_minutes_seconds(self) -> None:
        from app.date_utils import format_duration

        assert format_duration(3723.0) == "1h 2m 3s"

    def test_zero(self) -> None:
        from app.date_utils import format_duration

        assert format_duration(0.0) == "0s"

    def test_negative_raises(self) -> None:
        from app.date_utils import format_duration

        with pytest.raises(ValueError, match="non-negative"):
            format_duration(-1.0)

    @pytest.mark.parametrize("seconds,expected", [(60.0, "1m 0s"), (3600.0, "1h 0m 0s"), (86400.0, "24h 0m 0s")])
    def test_boundary_values(self, seconds: float, expected: str) -> None:
        from app.date_utils import format_duration

        assert format_duration(seconds) == expected


class TestIsLeapYear:
    def test_known_leap_year(self) -> None:
        from app.date_utils import is_leap_year

        assert is_leap_year(2024) is True

    def test_known_non_leap(self) -> None:
        from app.date_utils import is_leap_year

        assert is_leap_year(2023) is False

    def test_century_not_leap(self) -> None:
        from app.date_utils import is_leap_year

        assert is_leap_year(1900) is False

    def test_400_year_leap(self) -> None:
        from app.date_utils import is_leap_year

        assert is_leap_year(2000) is True

    def test_zero_raises(self) -> None:
        from app.date_utils import is_leap_year

        with pytest.raises(ValueError, match="positive"):
            is_leap_year(0)

    @pytest.mark.parametrize("year,expected", [(2020, True), (2100, False), (2000, True), (1999, False)])
    def test_parametrized(self, year: int, expected: bool) -> None:
        from app.date_utils import is_leap_year

        assert is_leap_year(year) == expected


class TestDaysInMonth:
    def test_january(self) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2026, 1) == 31

    def test_february_non_leap(self) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2023, 2) == 28

    def test_february_leap(self) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2024, 2) == 29

    def test_april_30_days(self) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2026, 4) == 30

    def test_invalid_month_raises(self) -> None:
        from app.date_utils import days_in_month

        with pytest.raises(ValueError, match="1-12"):
            days_in_month(2026, 13)

    @pytest.mark.parametrize("month,days", [(1, 31), (3, 31), (6, 30), (9, 30), (12, 31)])
    def test_standard_months(self, month: int, days: int) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2026, month) == days


class TestNextBusinessDay:
    def test_monday_returns_tuesday(self) -> None:
        from datetime import datetime

        from app.date_utils import next_business_day

        monday = datetime(2026, 8, 3)  # Monday
        result = next_business_day(monday)
        assert result.weekday() == 1  # Tuesday

    def test_friday_returns_monday(self) -> None:
        from datetime import datetime

        from app.date_utils import next_business_day

        friday = datetime(2026, 8, 7)  # Friday
        result = next_business_day(friday)
        assert result.weekday() == 0  # Monday

    def test_saturday_returns_monday(self) -> None:
        from datetime import datetime

        from app.date_utils import next_business_day

        saturday = datetime(2026, 8, 8)  # Saturday
        result = next_business_day(saturday)
        assert result.weekday() == 0  # Monday

    def test_sunday_returns_monday(self) -> None:
        from datetime import datetime

        from app.date_utils import next_business_day

        sunday = datetime(2026, 8, 9)  # Sunday
        result = next_business_day(sunday)
        assert result.weekday() == 0  # Monday


def test_quarter_of_year_q1() -> None:
    from app.date_utils import quarter_of_year

    assert quarter_of_year(datetime(2026, 1, 15)) == 1
    assert quarter_of_year(datetime(2026, 3, 31)) == 1


def test_quarter_of_year_q2() -> None:
    from app.date_utils import quarter_of_year

    assert quarter_of_year(datetime(2026, 4, 1)) == 2


def test_quarter_of_year_q4() -> None:
    from app.date_utils import quarter_of_year

    assert quarter_of_year(datetime(2026, 12, 31)) == 4


@pytest.mark.parametrize("month,expected_q", [(1, 1), (3, 1), (4, 2), (6, 2), (7, 3), (9, 3), (10, 4), (12, 4)])
def test_quarter_parametrize(month, expected_q) -> None:
    from app.date_utils import quarter_of_year

    assert quarter_of_year(datetime(2026, month, 1)) == expected_q


def test_is_weekend_saturday() -> None:
    from app.date_utils import is_weekend

    sat = datetime(2026, 8, 1)  # Saturday
    assert is_weekend(sat) is True


def test_is_weekend_sunday() -> None:
    from app.date_utils import is_weekend

    sun = datetime(2026, 8, 2)  # Sunday
    assert is_weekend(sun) is True


def test_is_weekend_monday() -> None:
    from app.date_utils import is_weekend

    mon = datetime(2026, 8, 3)  # Monday
    assert is_weekend(mon) is False


def test_days_until_future() -> None:
    from app.date_utils import days_until

    dt = datetime(2026, 1, 1)
    target = datetime(2026, 1, 11)
    assert days_until(dt, target) == 10


def test_days_until_past() -> None:
    from app.date_utils import days_until

    dt = datetime(2026, 1, 11)
    target = datetime(2026, 1, 1)
    assert days_until(dt, target) == -10


def test_days_until_same_day() -> None:
    from app.date_utils import days_until

    dt = datetime(2026, 6, 15)
    assert days_until(dt, dt) == 0


def test_format_iso_basic() -> None:
    from app.date_utils import format_iso

    dt = datetime(2026, 8, 6, 10, 30, 0)
    assert format_iso(dt) == "2026-08-06T10:30:00"


def test_format_iso_midnight() -> None:
    from app.date_utils import format_iso

    dt = datetime(2026, 1, 1, 0, 0, 0)
    assert format_iso(dt) == "2026-01-01T00:00:00"


@pytest.mark.parametrize("month,day", [(1, 1), (6, 15), (12, 31)])
def test_format_iso_various_dates(month, day) -> None:
    from app.date_utils import format_iso

    dt = datetime(2026, month, day, 12, 0, 0)
    iso = format_iso(dt)
    assert "T" in iso
    assert iso.startswith("2026")


def test_start_of_month_basic() -> None:
    dt = datetime(2024, 6, 15, 14, 30, 0)
    result = start_of_month(dt)
    assert result.day == 1
    assert result.hour == 0
    assert result.minute == 0
    assert result.month == 6
    assert result.year == 2024


def test_start_of_month_preserves_year() -> None:
    dt = datetime(2023, 12, 31, 23, 59, 59)
    result = start_of_month(dt)
    assert result.year == 2023
    assert result.month == 12
    assert result.day == 1


def test_start_of_month_already_first() -> None:
    dt = datetime(2024, 3, 1, 0, 0, 0)
    assert start_of_month(dt) == dt


@pytest.mark.parametrize("month", [1, 3, 6, 9, 12])
def test_start_of_month_parametrize(month: int) -> None:
    dt = datetime(2024, month, 15, 10, 0, 0)
    result = start_of_month(dt)
    assert result.month == month
    assert result.day == 1


def test_clamp_to_range_within() -> None:
    lo = datetime(2024, 1, 1)
    hi = datetime(2024, 12, 31)
    mid = datetime(2024, 6, 15)
    assert clamp_to_range(mid, lo, hi) == mid


def test_clamp_to_range_below() -> None:
    lo = datetime(2024, 6, 1)
    hi = datetime(2024, 12, 31)
    before = datetime(2024, 1, 1)
    assert clamp_to_range(before, lo, hi) == lo


def test_clamp_to_range_above() -> None:
    lo = datetime(2024, 1, 1)
    hi = datetime(2024, 6, 30)
    after = datetime(2024, 12, 31)
    assert clamp_to_range(after, lo, hi) == hi


def test_clamp_to_range_invalid_range_raises() -> None:
    lo = datetime(2024, 12, 31)
    hi = datetime(2024, 1, 1)
    with pytest.raises(ValueError):
        clamp_to_range(datetime(2024, 6, 15), lo, hi)


def test_clamp_to_range_equal_bounds() -> None:
    bound = datetime(2024, 6, 15)
    assert clamp_to_range(bound, bound, bound) == bound


# --- Tests for new date_utils functions ---


class TestWeekNumber:
    def test_first_week(self) -> None:
        from app.date_utils import week_number

        assert week_number(datetime(2024, 1, 1)) >= 1

    def test_returns_int(self) -> None:
        from app.date_utils import week_number

        assert isinstance(week_number(datetime(2024, 6, 15)), int)

    @pytest.mark.parametrize(
        "dt,expected_min,expected_max",
        [
            (datetime(2024, 1, 1), 1, 53),
            (datetime(2024, 12, 31), 1, 53),
        ],
    )
    def test_range(self, dt: datetime, expected_min: int, expected_max: int) -> None:
        from app.date_utils import week_number

        assert expected_min <= week_number(dt) <= expected_max


class TestFiscalQuarter:
    def test_calendar_q1(self) -> None:
        from app.date_utils import fiscal_quarter

        assert fiscal_quarter(datetime(2024, 2, 15)) == 1

    def test_calendar_q4(self) -> None:
        from app.date_utils import fiscal_quarter

        assert fiscal_quarter(datetime(2024, 12, 1)) == 4

    def test_fiscal_year_offset(self) -> None:
        from app.date_utils import fiscal_quarter

        # Fiscal year starts in April (UK-style)
        assert fiscal_quarter(datetime(2024, 4, 1), fiscal_year_start_month=4) == 1
        assert fiscal_quarter(datetime(2024, 7, 1), fiscal_year_start_month=4) == 2

    def test_invalid_month_raises(self) -> None:
        from app.date_utils import fiscal_quarter

        with pytest.raises(ValueError):
            fiscal_quarter(datetime(2024, 1, 1), fiscal_year_start_month=13)


class TestDaysInMonth:
    def test_february_non_leap(self) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2023, 2) == 28

    def test_february_leap(self) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2024, 2) == 29

    def test_thirty_one_day_month(self) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2024, 1) == 31

    def test_thirty_day_month(self) -> None:
        from app.date_utils import days_in_month

        assert days_in_month(2024, 4) == 30

    def test_invalid_month_raises(self) -> None:
        from app.date_utils import days_in_month

        with pytest.raises(ValueError):
            days_in_month(2024, 0)


class TestIsLeapYear:
    @pytest.mark.parametrize("year,expected", [(2024, True), (2023, False), (1900, False), (2000, True)])
    def test_parametrized(self, year: int, expected: bool) -> None:
        from app.date_utils import is_leap_year

        assert is_leap_year(year) == expected


class TestDateRangeOverlapDays:
    def test_no_overlap(self) -> None:
        from app.date_utils import date_range_overlap_days

        a_start, a_end = datetime(2024, 1, 1), datetime(2024, 1, 10)
        b_start, b_end = datetime(2024, 2, 1), datetime(2024, 2, 28)
        assert date_range_overlap_days(a_start, a_end, b_start, b_end) == 0

    def test_full_overlap(self) -> None:
        from app.date_utils import date_range_overlap_days

        d = datetime(2024, 6, 1)
        assert date_range_overlap_days(d, d, d, d) == 1

    def test_partial_overlap(self) -> None:
        from app.date_utils import date_range_overlap_days

        assert date_range_overlap_days(
            datetime(2024, 1, 1), datetime(2024, 1, 15),
            datetime(2024, 1, 10), datetime(2024, 1, 20),
        ) == 6

    def test_invalid_range_a_raises(self) -> None:
        from app.date_utils import date_range_overlap_days

        with pytest.raises(ValueError, match="Range A"):
            date_range_overlap_days(
                datetime(2024, 6, 30), datetime(2024, 6, 1),
                datetime(2024, 6, 1), datetime(2024, 6, 30),
            )
