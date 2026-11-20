"""Tests for fleet-mind app/warranty_tracker.py."""

from __future__ import annotations

from datetime import date

import pytest

from app.warranty_tracker import (
    WarrantyRecord,
    active_warranties,
    expiring_soon,
    warranty_coverage_rate,
)


def _record(
    vid: str = "v1",
    start: date = date(2024, 1, 1),
    expiry: date = date(2027, 1, 1),
    max_km: float | None = None,
    current_km: float = 0.0,
) -> WarrantyRecord:
    return WarrantyRecord(vid, "engine", start, expiry, max_km, current_km)


class TestIsExpiredByDate:
    def test_not_expired(self) -> None:
        r = _record(expiry=date(2030, 1, 1))
        assert not r.is_expired_by_date(date(2026, 11, 19))

    def test_expired(self) -> None:
        r = _record(expiry=date(2025, 1, 1))
        assert r.is_expired_by_date(date(2026, 11, 19))


class TestIsExpiredByKm:
    def test_no_max_km_not_expired(self) -> None:
        r = _record(max_km=None, current_km=1_000_000.0)
        assert not r.is_expired_by_km()

    def test_over_max_km_expired(self) -> None:
        r = _record(max_km=100_000.0, current_km=150_000.0)
        assert r.is_expired_by_km()


class TestDaysRemaining:
    def test_future_positive(self) -> None:
        r = _record(expiry=date(2026, 12, 19))
        assert r.days_remaining(date(2026, 11, 19)) == 30

    def test_expired_negative(self) -> None:
        r = _record(expiry=date(2026, 11, 9))
        assert r.days_remaining(date(2026, 11, 19)) < 0


class TestExpiringSoon:
    def test_returns_records_within_window(self) -> None:
        today = date(2026, 11, 19)
        soon = _record(expiry=date(2026, 12, 1))
        later = _record(expiry=date(2027, 6, 1), vid="v2")
        result = expiring_soon([soon, later], within_days=30, today=today)
        assert len(result) == 1

    def test_excludes_already_expired(self) -> None:
        today = date(2026, 11, 19)
        expired = _record(expiry=date(2026, 10, 1))
        result = expiring_soon([expired], within_days=30, today=today)
        assert result == []


class TestActiveWarranties:
    def test_filters_expired(self) -> None:
        today = date(2026, 11, 19)
        active = _record(expiry=date(2027, 1, 1))
        expired = _record(expiry=date(2026, 1, 1), vid="v2")
        result = active_warranties([active, expired], today)
        assert len(result) == 1


class TestWarrantyCoverageRate:
    def test_all_active_returns_one(self) -> None:
        today = date(2026, 11, 19)
        records = [_record(expiry=date(2027, 1, 1)), _record(expiry=date(2028, 1, 1), vid="v2")]
        assert warranty_coverage_rate(records, today) == pytest.approx(1.0)

    def test_empty_returns_zero(self) -> None:
        assert warranty_coverage_rate([], date(2026, 11, 19)) == pytest.approx(0.0)
