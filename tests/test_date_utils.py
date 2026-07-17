"""Tests for app/date_utils.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_utc_now_is_aware():
    from app.date_utils import utc_now
    now = utc_now()
    assert now.tzinfo is not None


def test_round_to_hour_zeroes_minutes():
    from app.date_utils import round_to_hour
    dt = datetime(2026, 6, 15, 14, 47, 33)
    result = round_to_hour(dt)
    assert result.minute == 0
    assert result.second == 0
    assert result.hour == 14


def test_hours_between_same():
    from app.date_utils import hours_between
    dt = datetime(2026, 1, 1, 12)
    assert hours_between(dt, dt) == 0


def test_hours_between_24h():
    from app.date_utils import hours_between
    from datetime import timedelta
    start = datetime(2026, 1, 1, 0)
    end = start + timedelta(hours=24)
    assert hours_between(start, end) == 24


def test_iso_to_datetime_basic():
    from app.date_utils import iso_to_datetime
    result = iso_to_datetime("2026-06-15T14:30:00")
    assert result.year == 2026
    assert result.hour == 14


def test_iso_to_datetime_with_z():
    from app.date_utils import iso_to_datetime
    result = iso_to_datetime("2026-06-15T14:30:00Z")
    assert result.tzinfo is not None


def test_iso_to_datetime_invalid_raises():
    from app.date_utils import iso_to_datetime
    with pytest.raises(ValueError, match="Cannot parse"):
        iso_to_datetime("not-a-date")


def test_generate_hourly_timestamps_length():
    from app.date_utils import generate_hourly_timestamps
    start = datetime(2026, 1, 1)
    result = generate_hourly_timestamps(start, 24)
    assert len(result) == 24


def test_generate_hourly_timestamps_spacing():
    from app.date_utils import generate_hourly_timestamps
    from datetime import timedelta
    start = datetime(2026, 1, 1)
    result = generate_hourly_timestamps(start, 3)
    assert result[1] - result[0] == timedelta(hours=1)


def test_generate_hourly_timestamps_zero_raises():
    from app.date_utils import generate_hourly_timestamps
    with pytest.raises(ValueError):
        generate_hourly_timestamps(datetime(2026, 1, 1), 0)


def test_is_business_hour_true():
    from app.date_utils import is_business_hour
    dt = datetime(2026, 6, 15, 10, 0)  # Monday 10am
    assert is_business_hour(dt) is True


def test_is_business_hour_false_weekend():
    from app.date_utils import is_business_hour
    dt = datetime(2026, 6, 14, 10, 0)  # Sunday
    assert is_business_hour(dt) is False


def test_is_business_hour_false_early():
    from app.date_utils import is_business_hour
    dt = datetime(2026, 6, 15, 7, 0)  # Monday 7am
    assert is_business_hour(dt) is False


def test_week_of_year():
    from app.date_utils import week_of_year
    dt = datetime(2026, 1, 5)  # First week
    assert 1 <= week_of_year(dt) <= 2


@pytest.mark.parametrize("n_hours", [1, 6, 24, 168])
def test_generate_hourly_timestamps_various_lengths(n_hours):
    from app.date_utils import generate_hourly_timestamps
    result = generate_hourly_timestamps(datetime(2026, 1, 1), n_hours)
    assert len(result) == n_hours
