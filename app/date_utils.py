"""Date and time utility helpers for Watt-Guard."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=UTC)


def round_to_hour(dt: datetime) -> datetime:
    """Round *dt* down to the nearest hour.

    Args:
        dt: Input datetime (any timezone).

    Returns:
        Datetime with minutes, seconds, and microseconds zeroed out.
    """
    return dt.replace(minute=0, second=0, microsecond=0)


def hours_between(start: datetime, end: datetime) -> int:
    """Return the number of whole hours between *start* and *end*.

    Args:
        start: Earlier datetime.
        end: Later datetime.

    Returns:
        Non-negative integer count of whole hours; negative if end < start.
    """
    delta = end - start
    return int(delta.total_seconds() // 3600)


def iso_to_datetime(iso_str: str) -> datetime:
    """Parse an ISO-8601 string to a datetime object.

    Args:
        iso_str: ISO-8601 formatted datetime string (e.g. '2026-06-15T14:00:00Z').

    Returns:
        Parsed datetime; naive if no timezone info is present.

    Raises:
        ValueError: If the string cannot be parsed as an ISO-8601 datetime.
    """
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        logger.debug("iso_to_datetime: parse failed for %r", iso_str)
        raise ValueError(f"Cannot parse ISO-8601 datetime: {iso_str!r}") from exc


def generate_hourly_timestamps(start: datetime, n_hours: int) -> list[datetime]:
    """Generate a sequence of *n_hours* hourly datetime objects starting from *start*.

    Args:
        start: Starting datetime.
        n_hours: Number of hourly timestamps to generate (must be >= 1).

    Returns:
        List of *n_hours* datetime objects spaced one hour apart.

    Raises:
        ValueError: If *n_hours* is less than 1.
    """
    if n_hours < 1:
        raise ValueError(f"n_hours must be at least 1, got {n_hours}")
    return [start + timedelta(hours=i) for i in range(n_hours)]


def is_business_hour(dt: datetime) -> bool:
    """Return True if *dt* falls within standard business hours (Mon-Fri, 08:00-18:00).

    Args:
        dt: Datetime to test (any timezone; local time fields are used).

    Returns:
        True when Monday-Friday and 8:00 <= hour < 18:00.
    """
    return dt.weekday() < 5 and 8 <= dt.hour < 18


def week_of_year(dt: datetime) -> int:
    """Return the ISO week number for *dt* (1-53).

    Args:
        dt: Any datetime.

    Returns:
        Integer ISO week number.
    """
    return dt.isocalendar()[1]


__all__ = [
    "business_days_between",
    "clamp_to_range",
    "date_range",
    "datetime_to_iso",
    "days_between",
    "days_until",
    "elapsed_seconds",
    "format_iso",
    "generate_hourly_timestamps",
    "hours_between",
    "is_business_day",
    "is_business_hour",
    "is_weekend",
    "iso_to_datetime",
    "next_midnight",
    "quarter_of_year",
    "round_to_hour",
    "start_of_day",
    "start_of_month",
    "utc_now",
    "week_of_year",
]


def datetime_to_iso(dt: datetime) -> str:
    """Serialise *dt* to an ISO-8601 string with UTC offset.

    Args:
        dt: Datetime to serialise (naive datetimes are assumed UTC).

    Returns:
        ISO-8601 string, e.g. '2026-08-03T12:00:00+00:00'.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def start_of_day(dt: datetime) -> datetime:
    """Return midnight (00:00:00) of the same day as *dt*.

    Args:
        dt: Any datetime.

    Returns:
        Datetime with time zeroed to midnight, preserving tzinfo.
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def days_between(start: datetime, end: datetime) -> int:
    """Return the number of whole calendar days between *start* and *end*.

    Args:
        start: Earlier datetime.
        end: Later datetime.

    Returns:
        Integer day count (can be negative if end < start).
    """
    delta = end - start
    return delta.days


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds (non-negative).

    Returns:
        Human-readable string such as '2h 5m 3s', '45m 0s', or '12s'.

    Raises:
        ValueError: If *seconds* is negative.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@lru_cache(maxsize=256)
def is_leap_year(year: int) -> bool:
    """Return True if *year* is a leap year.

    Args:
        year: Calendar year (positive integer).

    Returns:
        True when the year has 366 days, False otherwise.

    Raises:
        ValueError: If *year* is not a positive integer.
    """
    if year <= 0:
        raise ValueError(f"year must be positive, got {year}")
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


@lru_cache(maxsize=256)
def days_in_month(year: int, month: int) -> int:
    """Return the number of days in *month* of *year*.

    Args:
        year: Calendar year.
        month: Month number 1-12.

    Returns:
        Number of days in the given month.

    Raises:
        ValueError: If *month* is not in 1-12.
    """
    if month < 1 or month > 12:
        raise ValueError(f"month must be 1-12, got {month}")
    import calendar

    return calendar.monthrange(year, month)[1]


def next_business_day(dt: datetime) -> datetime:
    """Return the next business day (Monday-Friday) after *dt*.

    If *dt* falls on a weekday, returns the next weekday.
    If *dt* falls on Saturday, returns Monday.
    If *dt* falls on Sunday, returns Monday.

    Args:
        dt: Reference datetime.

    Returns:
        Datetime at midnight of the next business day, same tzinfo as *dt*.
    """
    import datetime as _dt

    d = dt.date() + _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    return datetime.combine(d, datetime.min.time()).replace(tzinfo=dt.tzinfo)


def quarter_of_year(dt: datetime) -> int:
    """Return the calendar quarter (1-4) for *dt*.

    Args:
        dt: Any datetime.

    Returns:
        Integer 1, 2, 3, or 4.
    """
    return (dt.month - 1) // 3 + 1


def is_weekend(dt: datetime) -> bool:
    """Return True if *dt* falls on a Saturday or Sunday.

    Args:
        dt: Any datetime.

    Returns:
        True when weekday is 5 (Saturday) or 6 (Sunday).
    """
    return dt.weekday() >= 5


def days_until(dt: datetime, target: datetime) -> int:
    """Return the number of whole days from *dt* to *target*.

    Negative when *target* is in the past relative to *dt*.

    Args:
        dt: Reference datetime.
        target: Target datetime.

    Returns:
        Integer number of days (may be negative).
    """
    return (target.date() - dt.date()).days


def format_iso(dt: datetime) -> str:
    """Format *dt* as an ISO 8601 string with second precision.

    Args:
        dt: Any datetime.

    Returns:
        String like '2024-06-15T14:30:00'.
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def start_of_month(dt: datetime) -> datetime:
    """Return a datetime at midnight on the first day of *dt*'s month.

    Args:
        dt: Any datetime (timezone is preserved).

    Returns:
        Datetime with day=1, hour=0, minute=0, second=0, microsecond=0.
    """
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def clamp_to_range(dt: datetime, start: datetime, end: datetime) -> datetime:
    """Clamp *dt* to the closed interval [start, end].

    Args:
        dt: Datetime to clamp.
        start: Lower bound (inclusive).
        end: Upper bound (inclusive).

    Returns:
        *start* if dt < start, *end* if dt > end, else *dt* unchanged.

    Raises:
        ValueError: If *start* > *end*.
    """
    if start > end:
        raise ValueError(f"start must not be after end: {start} > {end}")
    if dt < start:
        return start
    if dt > end:
        return end
    return dt


def date_range(start: datetime, end: datetime, step_hours: int = 24) -> list[datetime]:
    """Generate a list of datetimes from *start* to *end* (exclusive) at *step_hours* intervals.

    Args:
        start: Range start (inclusive).
        end: Range end (exclusive).
        step_hours: Interval between successive datetimes in hours (default 24).

    Returns:
        List of datetimes beginning at *start* and stepping by *step_hours* until *end*.

    Raises:
        ValueError: If *step_hours* < 1 or *start* >= *end*.
    """
    if step_hours < 1:
        raise ValueError(f"step_hours must be at least 1, got {step_hours}")
    if start >= end:
        raise ValueError("start must be before end")
    result: list[datetime] = []
    current = start
    delta = timedelta(hours=step_hours)
    while current < end:
        result.append(current)
        current += delta
    return result


def elapsed_seconds(start: datetime, end: datetime) -> float:
    """Return the number of seconds between *start* and *end*.

    Args:
        start: Earlier datetime.
        end: Later datetime.

    Returns:
        Total seconds as a float; negative when *end* precedes *start*.
    """
    return (end - start).total_seconds()


def next_midnight(dt: datetime) -> datetime:
    """Return the next midnight after *dt* (start of the following day).

    Args:
        dt: Reference datetime (timezone is preserved).

    Returns:
        Datetime set to midnight at the start of the next calendar day.
    """
    next_day = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return next_day


def business_days_between(start: datetime, end: datetime) -> int:
    """Count the number of business days (Mon-Fri) between *start* and *end*.

    Args:
        start: Start datetime (inclusive).
        end: End datetime (exclusive).

    Returns:
        Number of weekdays in [start.date(), end.date()).
    """
    if end <= start:
        return 0
    s = start.date()
    e = end.date()
    total = 0
    current = s
    while current < e:
        if current.weekday() < 5:
            total += 1
        current += timedelta(days=1)
    return total


def is_business_day(dt: datetime) -> bool:
    """Return True if *dt* falls on a Monday through Friday.

    Args:
        dt: Datetime to check.

    Returns:
        True for weekdays, False for Saturday and Sunday.
    """
    return dt.weekday() < 5
