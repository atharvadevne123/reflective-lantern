"""Tests for app.smart_meter_api module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.smart_meter_api import (
    DailyConsumption,
    MeterReading,
    aggregate_to_daily,
    consumption_stats,
    fill_missing_intervals,
    parse_hh_readings,
)


def _make_reading(hour: int, kwh: float, day: str = "2024-01-01") -> MeterReading:
    dt = datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00")
    return MeterReading(timestamp=dt, kwh=kwh)


def _make_raw(n: int = 4) -> list[dict[str, object]]:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    return [
        {"timestamp": (base + timedelta(minutes=30 * i)).isoformat(), "value": float(i)}
        for i in range(n)
    ]


class TestParseHHReadings:
    def test_basic_parse(self) -> None:
        raw = _make_raw(4)
        readings = parse_hh_readings(raw)
        assert len(readings) == 4

    def test_returns_sorted(self) -> None:
        raw = list(reversed(_make_raw(4)))
        readings = parse_hh_readings(raw)
        ts = [r.timestamp for r in readings]
        assert ts == sorted(ts)

    def test_missing_timestamp_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_hh_readings([{"value": 1.0}])

    def test_missing_value_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_hh_readings([{"timestamp": "2024-01-01T00:00:00+00:00"}])

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_hh_readings([{"timestamp": "2024-01-01T00:00:00+00:00", "value": -1.0}])

    def test_unix_timestamp(self) -> None:
        raw = [{"timestamp": 1704067200, "value": 5.0}]
        readings = parse_hh_readings(raw)
        assert len(readings) == 1

    @pytest.mark.parametrize("n", [1, 10, 48])
    def test_parametrized_counts(self, n: int) -> None:
        readings = parse_hh_readings(_make_raw(n))
        assert len(readings) == n


class TestAggregateToDaily:
    def test_single_day(self) -> None:
        readings = [_make_reading(h, 1.0) for h in range(24)]
        result = aggregate_to_daily(readings)
        assert len(result) == 1
        assert isinstance(result[0], DailyConsumption)

    def test_total_kwh(self) -> None:
        readings = [_make_reading(h, 2.0) for h in range(24)]
        result = aggregate_to_daily(readings)
        assert result[0].total_kwh == pytest.approx(48.0)

    def test_multi_day(self) -> None:
        r1 = [_make_reading(h, 1.0, "2024-01-01") for h in range(24)]
        r2 = [_make_reading(h, 2.0, "2024-01-02") for h in range(24)]
        result = aggregate_to_daily(r1 + r2)
        assert len(result) == 2

    def test_empty_returns_empty(self) -> None:
        assert aggregate_to_daily([]) == []

    @pytest.mark.parametrize("kwh_per_reading", [0.5, 1.0, 3.0])
    def test_off_peak_plus_peak_equals_total(self, kwh_per_reading: float) -> None:
        readings = [_make_reading(h, kwh_per_reading) for h in range(24)]
        result = aggregate_to_daily(readings)
        r = result[0]
        assert r.total_kwh == pytest.approx(r.peak_kwh + r.off_peak_kwh, rel=1e-5)


class TestFillMissingIntervals:
    def test_no_gaps_unchanged_length(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        readings = [MeterReading(base + timedelta(minutes=30 * i), float(i)) for i in range(4)]
        result = fill_missing_intervals(readings)
        assert len(result) == 4

    def test_single_gap_filled(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        readings = [
            MeterReading(base, 1.0),
            MeterReading(base + timedelta(hours=1), 2.0),
        ]
        result = fill_missing_intervals(readings)
        assert len(result) == 3

    def test_fill_value_used(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        readings = [
            MeterReading(base, 1.0),
            MeterReading(base + timedelta(hours=1), 2.0),
        ]
        result = fill_missing_intervals(readings, fill_value=0.0)
        assert result[1].kwh == pytest.approx(0.0)
        assert result[1].quality_flag == "estimated"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            fill_missing_intervals([])


class TestConsumptionStats:
    def test_basic_stats(self) -> None:
        readings = [_make_reading(h, float(h + 1)) for h in range(4)]
        stats = consumption_stats(readings)
        assert "total_kwh" in stats
        assert "mean_kwh" in stats

    def test_total_is_sum(self) -> None:
        readings = [_make_reading(h, 2.0) for h in range(5)]
        stats = consumption_stats(readings)
        assert stats["total_kwh"] == pytest.approx(10.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            consumption_stats([])

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_max_gte_min(self, n: int) -> None:
        readings = [_make_reading(i, float(i)) for i in range(n)]
        stats = consumption_stats(readings)
        assert stats["max_kwh"] >= stats["min_kwh"]
