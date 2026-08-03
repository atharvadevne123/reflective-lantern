"""Tests for app/date_utils.py."""

from __future__ import annotations

from datetime import datetime

import pytest


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
