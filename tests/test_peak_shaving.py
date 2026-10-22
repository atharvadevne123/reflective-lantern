"""Tests for app.peak_shaving module."""

from __future__ import annotations

import pytest

from app.peak_shaving import (
    BatteryState,
    PeakShavingResult,
    demand_charge_reduction,
    optimal_threshold,
    run_peak_shaving,
)


class TestBatteryState:
    def test_soc_fraction(self) -> None:
        b = BatteryState(100.0, 50.0, 50.0, 50.0)
        assert b.soc_fraction == pytest.approx(0.5)

    def test_available_kw_online_scenario(self) -> None:
        b = BatteryState(100.0, 80.0, 50.0, 50.0)
        b.status = "online"  # type: ignore[assignment]
        assert b.available_kw >= 0

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            BatteryState(0.0, 0.0, 10.0, 10.0)

    def test_soc_above_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            BatteryState(100.0, 110.0, 10.0, 10.0)

    def test_negative_soc_raises(self) -> None:
        with pytest.raises(ValueError):
            BatteryState(100.0, -10.0, 10.0, 10.0)

    def test_invalid_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            BatteryState(100.0, 50.0, 10.0, 10.0, round_trip_efficiency=0.0)

    def test_charge_increases_soc(self) -> None:
        b = BatteryState(100.0, 0.0, 50.0, 50.0)
        b.charge(50.0, 1.0)
        assert b.soc_kwh > 0

    def test_discharge_decreases_soc(self) -> None:
        b = BatteryState(100.0, 100.0, 50.0, 50.0)
        b.discharge(50.0, 1.0)
        assert b.soc_kwh < 100.0

    def test_charge_respects_capacity(self) -> None:
        b = BatteryState(100.0, 90.0, 100.0, 100.0)
        b.charge(100.0, 1.0)
        assert b.soc_kwh <= 100.0

    def test_discharge_respects_floor(self) -> None:
        b = BatteryState(100.0, 10.0, 100.0, 100.0)
        b.discharge(200.0, 1.0)
        assert b.soc_kwh >= 0.0


class TestRunPeakShaving:
    def _make_profile(self) -> list[float]:
        return [50.0, 80.0, 120.0, 100.0, 60.0, 40.0, 90.0, 110.0]

    def _make_battery(self) -> BatteryState:
        return BatteryState(50.0, 25.0, 30.0, 30.0)

    def test_returns_dataclass(self) -> None:
        result = run_peak_shaving(self._make_profile(), self._make_battery(), 100.0)
        assert isinstance(result, PeakShavingResult)

    def test_shaved_peak_not_above_original(self) -> None:
        result = run_peak_shaving(self._make_profile(), self._make_battery(), 100.0)
        assert result.shaved_peak_kw <= result.original_peak_kw

    def test_empty_profile_raises(self) -> None:
        with pytest.raises(ValueError):
            run_peak_shaving([], self._make_battery(), 100.0)

    def test_zero_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            run_peak_shaving(self._make_profile(), self._make_battery(), 0.0)

    def test_demand_charge_saved_non_negative(self) -> None:
        result = run_peak_shaving(self._make_profile(), self._make_battery(), 100.0)
        assert result.demand_charge_saved >= 0

    def test_interval_loads_length_matches_input(self) -> None:
        profile = self._make_profile()
        result = run_peak_shaving(profile, self._make_battery(), 100.0)
        assert len(result.interval_loads_kw) == len(profile)

    @pytest.mark.parametrize("threshold", [80.0, 100.0, 120.0])
    def test_various_thresholds(self, threshold: float) -> None:
        result = run_peak_shaving(self._make_profile(), self._make_battery(), threshold)
        assert result.shaved_peak_kw >= 0


class TestOptimalThreshold:
    def test_returns_float(self) -> None:
        t = optimal_threshold([50.0, 80.0, 120.0, 100.0], 50.0, 30.0)
        assert isinstance(t, float)

    def test_empty_profile_raises(self) -> None:
        with pytest.raises(ValueError):
            optimal_threshold([], 50.0, 30.0)

    def test_percentile_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            optimal_threshold([50.0, 80.0], 50.0, 30.0, percentile=101.0)

    def test_100th_percentile_is_max(self) -> None:
        profile = [10.0, 20.0, 30.0, 40.0, 50.0]
        t = optimal_threshold(profile, 50.0, 30.0, percentile=100.0)
        assert t == pytest.approx(50.0)


class TestDemandChargeReduction:
    def test_basic(self) -> None:
        saving = demand_charge_reduction(100.0, 80.0, 15.0)
        assert saving == pytest.approx(300.0)

    def test_no_reduction(self) -> None:
        assert demand_charge_reduction(100.0, 100.0, 15.0) == 0.0

    def test_shaved_above_original_raises(self) -> None:
        with pytest.raises(ValueError):
            demand_charge_reduction(80.0, 100.0, 15.0)

    @pytest.mark.parametrize(
        "orig,shaved,rate,expected",
        [
            (100.0, 80.0, 10.0, 200.0),
            (200.0, 150.0, 20.0, 1000.0),
        ],
    )
    def test_parametrized(self, orig: float, shaved: float, rate: float, expected: float) -> None:
        assert demand_charge_reduction(orig, shaved, rate) == pytest.approx(expected)
