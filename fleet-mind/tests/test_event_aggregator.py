"""Tests for fleet-mind app/event_aggregator.py."""

from __future__ import annotations

from datetime import date

import pytest

from app.event_aggregator import (
    FleetEvent,
    count_by_severity,
    count_by_type,
    critical_event_rate,
    daily_event_counts,
    events_in_window,
    events_per_vehicle,
)


def _ev(
    vid: str, etype: str = "sensor", severity: str = "low", occurred_on: date = date(2026, 11, 19)
) -> FleetEvent:
    return FleetEvent(vid, etype, severity, occurred_on)


class TestCountByType:
    def test_counts_each_type(self) -> None:
        events = [_ev("v1", "sensor"), _ev("v1", "sensor"), _ev("v2", "brake")]
        counts = count_by_type(events)
        assert counts["sensor"] == 2
        assert counts["brake"] == 1

    def test_empty_returns_empty(self) -> None:
        assert count_by_type([]) == {}


class TestCountBySeverity:
    def test_counts_severity(self) -> None:
        events = [_ev("v1", severity="low"), _ev("v2", severity="critical")]
        counts = count_by_severity(events)
        assert counts["low"] == 1
        assert counts["critical"] == 1


class TestEventsPerVehicle:
    def test_groups_by_vehicle(self) -> None:
        events = [_ev("v1"), _ev("v1"), _ev("v2")]
        counts = events_per_vehicle(events)
        assert counts["v1"] == 2
        assert counts["v2"] == 1


class TestEventsInWindow:
    def test_includes_boundary_dates(self) -> None:
        events = [
            _ev("v1", occurred_on=date(2026, 11, 1)),
            _ev("v2", occurred_on=date(2026, 11, 15)),
            _ev("v3", occurred_on=date(2026, 11, 30)),
        ]
        result = events_in_window(events, date(2026, 11, 1), date(2026, 11, 15))
        assert len(result) == 2

    def test_excludes_outside(self) -> None:
        events = [_ev("v1", occurred_on=date(2026, 10, 1))]
        result = events_in_window(events, date(2026, 11, 1), date(2026, 11, 30))
        assert len(result) == 0


class TestCriticalEventRate:
    def test_all_critical(self) -> None:
        events = [_ev("v1", severity="critical")] * 4
        assert critical_event_rate(events) == pytest.approx(1.0)

    def test_none_critical(self) -> None:
        events = [_ev("v1", severity="low")] * 4
        assert critical_event_rate(events) == pytest.approx(0.0)

    def test_empty_returns_zero(self) -> None:
        assert critical_event_rate([]) == pytest.approx(0.0)


class TestDailyEventCounts:
    def test_groups_by_date(self) -> None:
        events = [
            _ev("v1", occurred_on=date(2026, 11, 19)),
            _ev("v2", occurred_on=date(2026, 11, 19)),
            _ev("v3", occurred_on=date(2026, 11, 20)),
        ]
        counts = daily_event_counts(events)
        assert counts["2026-11-19"] == 2
        assert counts["2026-11-20"] == 1
