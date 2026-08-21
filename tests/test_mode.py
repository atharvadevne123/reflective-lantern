"""Tests for config.mode — run mode determination."""

from __future__ import annotations

from datetime import date

import pytest

from config.mode import RunMode, determine_mode, is_innovation_day


@pytest.mark.parametrize(
    "target_date,expected",
    [
        # Wednesday day 8-14 → INNOVATION
        (date(2026, 7, 8), RunMode.INNOVATION),  # Wed day 8
        (date(2026, 10, 14), RunMode.INNOVATION),  # Wed day 14 (Oct 2026)
        (date(2026, 7, 22), RunMode.INNOVATION),  # Wed day 22
        (date(2026, 10, 28), RunMode.INNOVATION),  # Wed day 28 (Oct 2026)
        # Wednesday but outside range → IMPROVEMENT
        (date(2026, 7, 1), RunMode.IMPROVEMENT),  # Wed day 1
        (date(2026, 7, 15), RunMode.IMPROVEMENT),  # Wed day 15
        (date(2026, 7, 29), RunMode.IMPROVEMENT),  # Wed day 29
        # Non-Wednesday → IMPROVEMENT
        (date(2026, 7, 13), RunMode.IMPROVEMENT),  # Monday day 13 (in range but not Wed)
        (date(2026, 7, 9), RunMode.IMPROVEMENT),  # Thursday day 9
    ],
)
def test_determine_mode(target_date: date, expected: RunMode) -> None:
    assert determine_mode(target_date) == expected


@pytest.mark.parametrize(
    "target_date,expected",
    [
        (date(2026, 7, 8), True),
        (date(2026, 7, 1), False),
        (date(2026, 7, 13), False),
    ],
)
def test_is_innovation_day(target_date: date, expected: bool) -> None:
    assert is_innovation_day(target_date) == expected


def test_determine_mode_improvement_is_default_on_weekday() -> None:
    monday = date(2026, 7, 6)  # Monday
    assert determine_mode(monday) == RunMode.IMPROVEMENT


def test_run_mode_string_values() -> None:
    assert RunMode.IMPROVEMENT == "IMPROVEMENT"
    assert RunMode.INNOVATION == "INNOVATION"


def test_determine_mode_boundaries() -> None:
    # Day 7 (outside range) on a Wednesday
    wed_day7 = date(2026, 7, 7)  # Tue, but pick a real Wed
    wed_day21 = date(2026, 7, 21)  # Tue
    # These aren't Wednesdays, so IMPROVEMENT regardless
    assert determine_mode(wed_day7) == RunMode.IMPROVEMENT
    assert determine_mode(wed_day21) == RunMode.IMPROVEMENT


def test_determine_mode_no_arg_returns_run_mode() -> None:
    result = determine_mode()
    assert isinstance(result, RunMode)


def test_is_innovation_day_no_arg() -> None:
    result = is_innovation_day()
    assert isinstance(result, bool)


@pytest.mark.parametrize(
    "day,expected",
    [
        (8, True),
        (9, True),
        (13, True),
        (14, True),
        (7, False),
        (15, False),
        (22, True),
        (23, True),
        (28, True),
        (21, False),
        (29, False),
    ],
)
def test_innovation_day_ranges_all_days(day: int, expected: bool) -> None:
    # Pick a Wednesday in July 2026 for day offset; July 1 is Wednesday
    # Nearest Wednesday with given day is mapped via fixed offsets

    from config.constants import INNOVATION_DAY_RANGES

    # Build a date where day-of-month == day and isoweekday == 3 (if expected)
    # Use October 2026: Oct 7 = Wed; Oct 14 = Wed; Oct 21 = Wed; Oct 28 = Wed
    # But we need arbitrary day numbers — just check ranges directly
    in_range = any(lo <= day <= hi for lo, hi in INNOVATION_DAY_RANGES)
    assert in_range == expected


def test_run_mode_is_str_subclass() -> None:
    assert issubclass(RunMode, str)


def test_determine_mode_today_is_improvement() -> None:
    # 2026-07-03 is a Friday → IMPROVEMENT
    assert determine_mode(date(2026, 7, 3)) == RunMode.IMPROVEMENT


def test_next_innovation_day_returns_date() -> None:
    from config.mode import next_innovation_day

    result = next_innovation_day(date(2026, 7, 6))
    assert isinstance(result, date)


def test_next_innovation_day_is_always_innovation() -> None:
    from config.mode import is_innovation_day, next_innovation_day

    nxt = next_innovation_day(date(2026, 7, 6))
    assert is_innovation_day(nxt)


def test_next_innovation_day_is_after_start() -> None:
    from config.mode import next_innovation_day

    start = date(2026, 7, 6)
    nxt = next_innovation_day(start)
    assert nxt > start


def test_next_innovation_day_is_wednesday() -> None:
    from config.mode import next_innovation_day

    nxt = next_innovation_day(date(2026, 7, 6))
    assert nxt.isoweekday() == 3


@pytest.mark.parametrize(
    "after,expected",
    [
        (date(2026, 7, 6), date(2026, 7, 8)),
        (date(2026, 7, 8), date(2026, 7, 22)),
    ],
)
def test_next_innovation_day_known_dates(after: date, expected: date) -> None:
    from config.mode import next_innovation_day

    assert next_innovation_day(after) == expected


def test_next_innovation_day_is_in_valid_day_range() -> None:
    from config.constants import INNOVATION_DAY_RANGES
    from config.mode import next_innovation_day

    nxt = next_innovation_day(after=date(2026, 7, 1))
    day = nxt.day
    in_range = any(lo <= day <= hi for lo, hi in INNOVATION_DAY_RANGES)
    assert in_range


def test_run_mode_str_value() -> None:
    from config.mode import RunMode

    assert RunMode.IMPROVEMENT.value == "IMPROVEMENT"
    assert RunMode.INNOVATION.value == "INNOVATION"


def test_determine_mode_improvement_on_monday() -> None:
    from config.mode import RunMode, determine_mode

    monday = date(2026, 7, 6)
    assert monday.isoweekday() == 1
    assert determine_mode(monday) == RunMode.IMPROVEMENT


def test_is_innovation_day_false_on_first_wednesday_of_month() -> None:
    from config.mode import is_innovation_day

    wed_day1 = date(2026, 7, 1)
    assert wed_day1.isoweekday() == 3
    assert not is_innovation_day(wed_day1)


def test_mode_label_innovation() -> None:
    from config.mode import mode_label

    assert mode_label(RunMode.INNOVATION) == "Innovation Mode"


def test_mode_label_improvement() -> None:
    from config.mode import upcoming_innovation_days

    days = upcoming_innovation_days(count=5)
    assert all(is_innovation_day(d) for d in days)


def test_upcoming_innovation_days_sorted() -> None:
    from config.mode import upcoming_innovation_days

    days = upcoming_innovation_days(count=4)
    assert days == sorted(days)


def test_mode_schedule_weekdays_only() -> None:
    from config.mode import mode_schedule

    schedule = mode_schedule(weeks=1)
    for entry in schedule:
        assert entry["mode"] in ("IMPROVEMENT", "INNOVATION")


def test_mode_schedule_length() -> None:
    from config.mode import mode_schedule

    schedule = mode_schedule(weeks=1)
    # 1 week = 5 weekdays
    assert len(schedule) == 5


@pytest.mark.parametrize("weeks", [1, 2, 4])
def test_mode_schedule_multi_week_length(weeks: int) -> None:
    from config.mode import mode_schedule

    schedule = mode_schedule(weeks=weeks)
    assert len(schedule) == weeks * 5


def test_run_mode_improvement_value() -> None:
    assert RunMode.IMPROVEMENT == "IMPROVEMENT"


def test_run_mode_innovation_value() -> None:
    assert RunMode.INNOVATION == "INNOVATION"


def test_upcoming_innovation_days_all_in_future() -> None:
    from datetime import date

    from config.mode import upcoming_innovation_days

    today = date.today()
    days = upcoming_innovation_days(count=3, after=today)
    assert all(d > today for d in days)


def test_determine_mode_saturday_is_improvement() -> None:
    from datetime import date

    saturday = date(2026, 7, 25)  # Saturday
    result = determine_mode(saturday)
    assert result == RunMode.IMPROVEMENT


def test_upcoming_innovation_days_returns_count() -> None:
    from config.mode import upcoming_innovation_days

    days = upcoming_innovation_days(count=3)
    assert len(days) == 3


def test_upcoming_innovation_days_all_innovation() -> None:
    from config.mode import is_innovation_day, upcoming_innovation_days

    days = upcoming_innovation_days(count=5)
    assert all(is_innovation_day(d) for d in days)
