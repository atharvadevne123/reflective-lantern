"""Smart meter API data fetcher and normaliser for half-hourly interval data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

HALF_HOUR_SECONDS: int = 1800
INTERVALS_PER_DAY: int = 48


@dataclass
class MeterReading:
    """A single smart meter interval reading."""

    timestamp: datetime
    kwh: float
    quality_flag: str = "good"


@dataclass
class DailyConsumption:
    """Aggregated daily consumption summary."""

    date: str
    total_kwh: float
    peak_kwh: float
    off_peak_kwh: float
    readings: int


def parse_hh_readings(
    raw: list[dict[str, object]],
    timestamp_key: str = "timestamp",
    value_key: str = "value",
) -> list[MeterReading]:
    """Parse a list of raw half-hourly reading dicts into :class:`MeterReading` objects.

    Args:
        raw: List of dicts, each containing timestamp (ISO-8601 str or UNIX int) and value (kWh).
        timestamp_key: Key for the timestamp field.
        value_key: Key for the kWh value field.

    Returns:
        List of parsed :class:`MeterReading` objects, sorted by timestamp.

    Raises:
        ValueError: If a required key is missing or a value is negative.
    """
    readings: list[MeterReading] = []
    for i, item in enumerate(raw):
        if timestamp_key not in item:
            raise ValueError(f"Missing '{timestamp_key}' at index {i}")
        if value_key not in item:
            raise ValueError(f"Missing '{value_key}' at index {i}")
        ts = item[timestamp_key]
        val = float(item[value_key])  # type: ignore[arg-type]
        if val < 0:
            raise ValueError(f"Negative kWh value at index {i}: {val}")
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=UTC)
        else:
            dt = datetime.fromisoformat(str(ts))
        quality = "good" if val >= 0 else "negative"
        readings.append(MeterReading(timestamp=dt, kwh=val, quality_flag=quality))
    return sorted(readings, key=lambda r: r.timestamp)


def aggregate_to_daily(readings: list[MeterReading]) -> list[DailyConsumption]:
    """Aggregate half-hourly readings into daily consumption summaries.

    Peak hours are defined as 16:00–20:59 (hours 16–20 inclusive).

    Args:
        readings: List of :class:`MeterReading` objects (may span multiple days).

    Returns:
        List of :class:`DailyConsumption`, one per calendar day, sorted ascending.
    """
    daily: dict[str, list[MeterReading]] = {}
    for r in readings:
        day = r.timestamp.strftime("%Y-%m-%d")
        daily.setdefault(day, []).append(r)
    result: list[DailyConsumption] = []
    for day, day_readings in sorted(daily.items()):
        total = round(sum(r.kwh for r in day_readings), 4)
        peak = round(sum(r.kwh for r in day_readings if 16 <= r.timestamp.hour <= 20), 4)
        off_peak = round(total - peak, 4)
        result.append(
            DailyConsumption(
                date=day,
                total_kwh=total,
                peak_kwh=peak,
                off_peak_kwh=off_peak,
                readings=len(day_readings),
            )
        )
    return result


def fill_missing_intervals(
    readings: list[MeterReading],
    fill_value: float = 0.0,
) -> list[MeterReading]:
    """Insert zero-value readings for any missing half-hourly intervals.

    Args:
        readings: Sorted list of :class:`MeterReading` objects.
        fill_value: kWh value for inserted gap readings (default 0.0).

    Returns:
        New list with gaps filled.  Original readings are not mutated.

    Raises:
        ValueError: If readings is empty.
    """
    if not readings:
        raise ValueError("readings must not be empty")
    filled: list[MeterReading] = [readings[0]]
    from datetime import timedelta

    step = timedelta(seconds=HALF_HOUR_SECONDS)
    for prev, curr in zip(readings, readings[1:], strict=False):
        expected = prev.timestamp + step
        while expected < curr.timestamp:
            filled.append(
                MeterReading(timestamp=expected, kwh=fill_value, quality_flag="estimated")
            )
            expected += step
        filled.append(curr)
    return filled


def consumption_stats(readings: list[MeterReading]) -> dict[str, float]:
    """Compute basic descriptive statistics for a set of readings.

    Args:
        readings: List of :class:`MeterReading` objects.

    Returns:
        Dict with keys ``total_kwh``, ``mean_kwh``, ``max_kwh``, ``min_kwh``.

    Raises:
        ValueError: If readings is empty.
    """
    if not readings:
        raise ValueError("readings must not be empty")
    values = [r.kwh for r in readings]
    return {
        "total_kwh": round(sum(values), 4),
        "mean_kwh": round(sum(values) / len(values), 6),
        "max_kwh": max(values),
        "min_kwh": min(values),
    }
