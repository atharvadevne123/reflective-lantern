"""Tests for fleet-mind app/seasonal_adjustment.py."""

from __future__ import annotations

from datetime import date

import pytest

from app.seasonal_adjustment import (
    adjust_forecast,
    adjustment_factor,
    season_for_date,
    seasonal_maintenance_schedule,
)


class TestSeasonForDate:
    def test_winter_december(self) -> None:
        assert season_for_date(date(2026, 12, 15)) == "winter"

    def test_spring_april(self) -> None:
        assert season_for_date(date(2026, 4, 1)) == "spring"

    def test_summer_july(self) -> None:
        assert season_for_date(date(2026, 7, 1)) == "summer"

    def test_autumn_october(self) -> None:
        assert season_for_date(date(2026, 10, 1)) == "autumn"


class TestAdjustmentFactor:
    def test_spring_returns_one(self) -> None:
        assert adjustment_factor(date(2026, 4, 15)) == pytest.approx(1.0)

    def test_winter_above_one(self) -> None:
        assert adjustment_factor(date(2026, 1, 15)) > 1.0


class TestAdjustForecast:
    def test_scales_by_factor(self) -> None:
        # winter factor = 1.25, so 100.0 → 125.0
        result = adjust_forecast(100.0, date(2026, 1, 15))
        assert result == pytest.approx(125.0)

    def test_spring_unchanged(self) -> None:
        result = adjust_forecast(100.0, date(2026, 4, 15))
        assert result == pytest.approx(100.0)


class TestSeasonalMaintenanceSchedule:
    def test_returns_four_seasons(self) -> None:
        schedule = seasonal_maintenance_schedule(90, 2026)
        assert len(schedule) == 4

    def test_winter_interval_shorter_than_base(self) -> None:
        schedule = seasonal_maintenance_schedule(90, 2026)
        winter = next(s for s in schedule if s["season"] == "winter")
        assert winter["recommended_interval_days"] < 90

    def test_contains_required_keys(self) -> None:
        schedule = seasonal_maintenance_schedule(90, 2026)
        for s in schedule:
            assert "season" in s
            assert "adjustment_factor" in s
            assert "recommended_interval_days" in s
