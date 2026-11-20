"""Tests for fleet-mind app/driver_behavior.py."""

from __future__ import annotations

import pytest

from app.driver_behavior import (
    DrivingEvent,
    driver_score,
    harsh_event_rate,
    rank_drivers,
    severity_stats,
    speeding_fraction,
)


def _event(driver: str, etype: str, severity: float = 0.5, speed: float = 80.0) -> DrivingEvent:
    return DrivingEvent(driver_id=driver, event_type=etype, severity=severity, speed_kmh=speed)


class TestHarshEventRate:
    def test_no_harsh_events(self) -> None:
        events = [_event("d1", "normal")]
        assert harsh_event_rate(events, 100.0) == pytest.approx(0.0)

    def test_harsh_events_counted(self) -> None:
        events = [_event("d1", "harsh_brake"), _event("d1", "harsh_accel")]
        rate = harsh_event_rate(events, 100.0)
        assert rate == pytest.approx(2.0)

    def test_zero_distance_returns_zero(self) -> None:
        assert harsh_event_rate([_event("d1", "harsh_brake")], 0.0) == pytest.approx(0.0)


class TestSpeedingFraction:
    def test_no_speeding(self) -> None:
        events = [_event("d1", "normal", speed=80.0)]
        assert speeding_fraction(events, 100.0) == pytest.approx(0.0)

    def test_all_speeding(self) -> None:
        events = [_event("d1", "normal", speed=120.0)]
        assert speeding_fraction(events, 100.0) == pytest.approx(1.0)

    def test_empty_events(self) -> None:
        assert speeding_fraction([], 100.0) == pytest.approx(0.0)


class TestDriverScore:
    def test_perfect_driver(self) -> None:
        assert driver_score([], 100.0) == pytest.approx(100.0)

    def test_no_distance_returns_100(self) -> None:
        events = [_event("d1", "harsh_brake")]
        assert driver_score(events, 0.0) == pytest.approx(100.0)

    def test_score_clamped_at_zero(self) -> None:
        events = [_event("d1", "harsh_brake")] * 50
        score = driver_score(events, 1.0, speed_limit_kmh=50.0)
        assert score >= 0.0


class TestSeverityStats:
    def test_empty_events(self) -> None:
        result = severity_stats([])
        assert result["mean"] == 0.0

    def test_single_event(self) -> None:
        result = severity_stats([_event("d1", "t", severity=0.8)])
        assert result["mean"] == pytest.approx(0.8)
        assert result["std"] == pytest.approx(0.0)


class TestRankDrivers:
    def test_sorted_descending(self) -> None:
        scores = {"alice": 90.0, "bob": 70.0, "carol": 85.0}
        ranked = rank_drivers(scores)
        assert ranked[0][0] == "alice"
        assert ranked[-1][0] == "bob"
