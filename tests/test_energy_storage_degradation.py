"""Tests for app.energy_storage_degradation module."""

from __future__ import annotations

import pytest

from app.energy_storage_degradation import (
    EOL_CAPACITY_FRACTION,
    NOMINAL_CYCLES_LFP,
    DegradationResult,
    calendar_degradation,
    cycle_degradation,
    cycles_to_eol,
    degradation_report,
    rainflow_cycle_depth_factor,
    remaining_capacity,
)


class TestRainflowCycleDepthFactor:
    def test_full_dod_returns_one(self) -> None:
        assert rainflow_cycle_depth_factor(1.0) == pytest.approx(1.0)

    def test_partial_dod_less_than_one(self) -> None:
        factor = rainflow_cycle_depth_factor(0.5)
        assert factor < 1.0

    def test_zero_dod_raises(self) -> None:
        with pytest.raises(ValueError):
            rainflow_cycle_depth_factor(0.0)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            rainflow_cycle_depth_factor(1.1)

    @pytest.mark.parametrize("dod", [0.2, 0.5, 0.8, 1.0])
    def test_monotonically_increasing(self, dod: float) -> None:
        if dod < 1.0:
            assert rainflow_cycle_depth_factor(dod) < rainflow_cycle_depth_factor(1.0)


class TestCycleDegradation:
    def test_zero_cycles_no_degradation(self) -> None:
        assert cycle_degradation(0.0, NOMINAL_CYCLES_LFP) == 0.0

    def test_eol_at_rated_cycles(self) -> None:
        deg = cycle_degradation(
            float(NOMINAL_CYCLES_LFP), NOMINAL_CYCLES_LFP, depth_of_discharge=1.0
        )
        assert deg == pytest.approx(1.0 - EOL_CAPACITY_FRACTION, rel=1e-3)

    def test_negative_cycles_raises(self) -> None:
        with pytest.raises(ValueError):
            cycle_degradation(-1.0, NOMINAL_CYCLES_LFP)

    def test_zero_nominal_cycles_raises(self) -> None:
        with pytest.raises(ValueError):
            cycle_degradation(100.0, 0)

    def test_higher_dod_more_degradation(self) -> None:
        d_low = cycle_degradation(500.0, NOMINAL_CYCLES_LFP, depth_of_discharge=0.5)
        d_high = cycle_degradation(500.0, NOMINAL_CYCLES_LFP, depth_of_discharge=1.0)
        assert d_high > d_low


class TestCalendarDegradation:
    def test_zero_years_no_degradation(self) -> None:
        assert calendar_degradation(0.0) == 0.0

    def test_annual_rate_after_one_year(self) -> None:
        deg = calendar_degradation(1.0, degradation_per_year=0.02)
        assert deg == pytest.approx(0.02, rel=1e-4)

    def test_negative_years_raises(self) -> None:
        with pytest.raises(ValueError):
            calendar_degradation(-1.0)

    def test_zero_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            calendar_degradation(1.0, degradation_per_year=0.0)

    def test_temperature_factor_increases_degradation(self) -> None:
        d_normal = calendar_degradation(5.0, temperature_factor=1.0)
        d_hot = calendar_degradation(5.0, temperature_factor=2.0)
        assert d_hot > d_normal

    @pytest.mark.parametrize("years", [1.0, 5.0, 10.0])
    def test_monotonically_increasing(self, years: float) -> None:
        d1 = calendar_degradation(1.0)
        d = calendar_degradation(years)
        if years > 1.0:
            assert d >= d1


class TestRemainingCapacity:
    def test_no_degradation(self) -> None:
        assert remaining_capacity(100.0, 0.0, 0.0) == pytest.approx(100.0)

    def test_full_degradation_gives_zero(self) -> None:
        assert remaining_capacity(100.0, 0.5, 0.5) == pytest.approx(0.0)

    def test_zero_nameplate_raises(self) -> None:
        with pytest.raises(ValueError):
            remaining_capacity(0.0, 0.1, 0.1)

    def test_invalid_cycle_deg_raises(self) -> None:
        with pytest.raises(ValueError):
            remaining_capacity(100.0, -0.1, 0.1)

    def test_invalid_calendar_deg_raises(self) -> None:
        with pytest.raises(ValueError):
            remaining_capacity(100.0, 0.1, 1.1)


class TestCyclesToEOL:
    def test_new_battery_has_many_remaining(self) -> None:
        remaining = cycles_to_eol(NOMINAL_CYCLES_LFP, 0.0, 0.0)
        assert remaining > 0

    def test_exhausted_battery_returns_zero(self) -> None:
        remaining = cycles_to_eol(NOMINAL_CYCLES_LFP, float(NOMINAL_CYCLES_LFP * 2), 10.0)
        assert remaining == 0.0


class TestDegradationReport:
    def test_returns_dataclass(self) -> None:
        result = degradation_report(100.0, 500.0, 2.0)
        assert isinstance(result, DegradationResult)

    def test_fresh_battery_not_at_eol(self) -> None:
        result = degradation_report(100.0, 0.0, 0.0)
        assert not result.eol_reached

    def test_capacity_fraction_in_range(self) -> None:
        result = degradation_report(100.0, 500.0, 2.0)
        assert 0 <= result.capacity_fraction <= 1

    def test_capacity_kwh_positive(self) -> None:
        result = degradation_report(100.0, 500.0, 2.0)
        assert result.capacity_kwh >= 0

    def test_heavily_cycled_reaches_eol(self) -> None:
        result = degradation_report(100.0, float(NOMINAL_CYCLES_LFP * 3), 20.0)
        assert result.eol_reached

    @pytest.mark.parametrize("cycles", [0.0, 500.0, 1500.0])
    def test_increasing_cycles_reduces_capacity(self, cycles: float) -> None:
        r0 = degradation_report(100.0, 0.0, 0.0)
        rc = degradation_report(100.0, cycles, 0.0)
        assert rc.capacity_fraction <= r0.capacity_fraction
