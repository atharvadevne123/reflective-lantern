"""Tests for app.occupancy_model module."""

from __future__ import annotations

import pytest

from app.occupancy_model import (
    OFFICE_BASE_SCHEDULE,
    RESIDENTIAL_BASE_SCHEDULE,
    OccupancyProfile,
    annual_occupied_hours,
    apply_daytype_factor,
    build_occupancy_profile,
    energy_with_occupancy,
    occupancy_adjusted_setpoint,
)


class TestApplyDaytypeFactor:
    def test_weekday_unchanged(self) -> None:
        result = apply_daytype_factor(OFFICE_BASE_SCHEDULE, is_weekend=False)
        assert result == OFFICE_BASE_SCHEDULE

    def test_weekend_office_reduced(self) -> None:
        result = apply_daytype_factor(OFFICE_BASE_SCHEDULE, is_weekend=True)
        assert all(v <= OFFICE_BASE_SCHEDULE[h] for h, v in result.items())

    def test_weekend_fractions_capped_at_one(self) -> None:
        schedule = {h: 0.9 for h in range(24)}
        result = apply_daytype_factor(schedule, is_weekend=True)
        assert all(v <= 1.0 for v in result.values())


class TestBuildOccupancyProfile:
    def test_returns_occupancy_profile(self) -> None:
        result = build_occupancy_profile()
        assert isinstance(result, OccupancyProfile)

    def test_hourly_fractions_length(self) -> None:
        result = build_occupancy_profile()
        assert len(result.hourly_fractions) == 24

    def test_peak_fraction_is_max(self) -> None:
        result = build_occupancy_profile()
        assert result.peak_fraction == max(result.hourly_fractions)

    def test_average_fraction_within_bounds(self) -> None:
        result = build_occupancy_profile()
        assert 0.0 <= result.average_fraction <= 1.0

    def test_weekend_has_fewer_occupied_hours(self) -> None:
        weekday = build_occupancy_profile(is_weekend=False)
        weekend = build_occupancy_profile(is_weekend=True)
        assert weekend.occupied_hours <= weekday.occupied_hours

    def test_residential_schedule(self) -> None:
        result = build_occupancy_profile(schedule=RESIDENTIAL_BASE_SCHEDULE)
        assert result.average_fraction > 0.0

    @pytest.mark.parametrize("is_weekend", [True, False])
    def test_fractions_within_zero_one(self, is_weekend: bool) -> None:
        result = build_occupancy_profile(is_weekend=is_weekend)
        assert all(0.0 <= f <= 1.0 for f in result.hourly_fractions)


class TestOccupancyAdjustedSetpoint:
    def test_occupied_setpoint_when_above_threshold(self) -> None:
        assert occupancy_adjusted_setpoint(21.0, 16.0, 0.5) == 21.0

    def test_unoccupied_setpoint_when_below_threshold(self) -> None:
        assert occupancy_adjusted_setpoint(21.0, 16.0, 0.02) == 16.0

    def test_exactly_at_threshold_returns_occupied(self) -> None:
        assert occupancy_adjusted_setpoint(21.0, 16.0, 0.05) == 21.0

    def test_negative_fraction_raises(self) -> None:
        with pytest.raises(ValueError):
            occupancy_adjusted_setpoint(21.0, 16.0, -0.1)

    def test_fraction_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            occupancy_adjusted_setpoint(21.0, 16.0, 1.1)

    @pytest.mark.parametrize(
        "occupied,unoccupied,frac,expected",
        [(21.0, 16.0, 1.0, 21.0), (22.0, 15.0, 0.0, 15.0)],
    )
    def test_parametrized(
        self,
        occupied: float,
        unoccupied: float,
        frac: float,
        expected: float,
    ) -> None:
        assert occupancy_adjusted_setpoint(occupied, unoccupied, frac) == expected


class TestEnergyWithOccupancy:
    def test_full_occupancy_unchanged(self) -> None:
        base = [10.0] * 24
        fractions = [1.0] * 24
        result = energy_with_occupancy(base, fractions)
        assert result == pytest.approx(base)

    def test_zero_occupancy_reduces_consumption(self) -> None:
        base = [10.0] * 24
        fractions = [0.0] * 24
        result = energy_with_occupancy(base, fractions, 0.30)
        assert all(v < 10.0 for v in result)

    def test_unequal_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            energy_with_occupancy([1.0] * 24, [0.5] * 12)

    def test_invalid_reduction_factor_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_with_occupancy([1.0] * 24, [0.5] * 24, 1.0)

    @pytest.mark.parametrize("reduction", [0.0, 0.2, 0.5])
    def test_results_non_negative(self, reduction: float) -> None:
        base = [5.0] * 24
        fracs = [0.3] * 24
        result = energy_with_occupancy(base, fracs, reduction)
        assert all(v >= 0.0 for v in result)


class TestAnnualOccupiedHours:
    def test_all_day_every_day(self) -> None:
        full = {h: 1.0 for h in range(24)}
        hours = annual_occupied_hours(full, full)
        assert hours == 365 * 24

    def test_no_occupancy(self) -> None:
        empty = {h: 0.0 for h in range(24)}
        hours = annual_occupied_hours(empty)
        assert hours == 0

    def test_office_hours_reasonable(self) -> None:
        hours = annual_occupied_hours(OFFICE_BASE_SCHEDULE)
        assert 1000 < hours < 5000

    @pytest.mark.parametrize("weekdays", [260, 261, 262])
    def test_parametrized_weekdays(self, weekdays: int) -> None:
        hours = annual_occupied_hours(
            OFFICE_BASE_SCHEDULE, weekdays_per_year=weekdays
        )
        assert hours >= 0
