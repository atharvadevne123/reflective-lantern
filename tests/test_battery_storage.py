"""Tests for app.battery_storage module."""

from __future__ import annotations

import pytest

from app.battery_storage import (
    BatterySystem,
    DispatchResult,
    charge_kwh,
    discharge_kwh,
    peak_shaving_threshold,
    simulate_peak_shaving,
    usable_energy_kwh,
)


def _battery() -> BatterySystem:
    return BatterySystem(
        capacity_kwh=100.0,
        max_charge_kw=25.0,
        max_discharge_kw=25.0,
        round_trip_efficiency=0.90,
        initial_soc_pct=50.0,
    )


class TestUsableEnergyKwh:
    def test_half_soc(self) -> None:
        assert usable_energy_kwh(_battery(), 50.0) == pytest.approx(50.0)

    def test_full_soc(self) -> None:
        assert usable_energy_kwh(_battery(), 100.0) == pytest.approx(100.0)

    def test_zero_soc(self) -> None:
        assert usable_energy_kwh(_battery(), 0.0) == pytest.approx(0.0)

    def test_invalid_soc_raises(self) -> None:
        with pytest.raises(ValueError):
            usable_energy_kwh(_battery(), 110.0)

    @pytest.mark.parametrize("soc", [0.0, 25.0, 50.0, 75.0, 100.0])
    def test_parametrized(self, soc: float) -> None:
        result = usable_energy_kwh(_battery(), soc)
        assert 0.0 <= result <= 100.0


class TestChargeKwh:
    def test_basic_charge(self) -> None:
        grid_kwh, new_soc = charge_kwh(_battery(), 10.0, 1.0, 50.0)
        assert grid_kwh > 0.0
        assert new_soc > 50.0

    def test_soc_increases(self) -> None:
        _, new_soc = charge_kwh(_battery(), 10.0, 2.0, 30.0)
        assert new_soc > 30.0

    def test_zero_power_raises(self) -> None:
        with pytest.raises(ValueError):
            charge_kwh(_battery(), 0.0, 1.0, 50.0)

    def test_zero_hours_raises(self) -> None:
        with pytest.raises(ValueError):
            charge_kwh(_battery(), 10.0, 0.0, 50.0)

    def test_soc_capped_at_100(self) -> None:
        _, new_soc = charge_kwh(_battery(), 25.0, 10.0, 95.0)
        assert new_soc <= 100.0


class TestDischargeKwh:
    def test_basic_discharge(self) -> None:
        delivered, new_soc = discharge_kwh(_battery(), 10.0, 1.0, 80.0)
        assert delivered > 0.0
        assert new_soc < 80.0

    def test_cannot_discharge_below_zero(self) -> None:
        _, new_soc = discharge_kwh(_battery(), 25.0, 10.0, 5.0)
        assert new_soc >= 0.0

    def test_zero_power_raises(self) -> None:
        with pytest.raises(ValueError):
            discharge_kwh(_battery(), 0.0, 1.0, 50.0)

    def test_invalid_soc_raises(self) -> None:
        with pytest.raises(ValueError):
            discharge_kwh(_battery(), 10.0, 1.0, -5.0)

    @pytest.mark.parametrize("soc", [20.0, 50.0, 80.0])
    def test_soc_decreases(self, soc: float) -> None:
        _, new_soc = discharge_kwh(_battery(), 10.0, 1.0, soc)
        assert new_soc <= soc


class TestPeakShavingThreshold:
    def test_zero_reduction(self) -> None:
        profile = [10.0, 20.0, 30.0, 15.0]
        assert peak_shaving_threshold(profile, 0.0) == pytest.approx(30.0)

    def test_100_reduction(self) -> None:
        profile = [10.0, 20.0, 30.0]
        assert peak_shaving_threshold(profile, 100.0) == pytest.approx(0.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            peak_shaving_threshold([], 10.0)

    def test_invalid_target_raises(self) -> None:
        with pytest.raises(ValueError):
            peak_shaving_threshold([10.0, 20.0], 110.0)

    @pytest.mark.parametrize("pct", [10.0, 20.0, 30.0])
    def test_threshold_below_peak(self, pct: float) -> None:
        profile = [50.0, 100.0, 75.0]
        threshold = peak_shaving_threshold(profile, pct)
        assert threshold < 100.0


class TestSimulatePeakShaving:
    def test_returns_dispatch_result(self) -> None:
        profile = [20.0, 30.0, 50.0, 25.0, 15.0]
        result = simulate_peak_shaving(_battery(), profile, 35.0)
        assert isinstance(result, DispatchResult)

    def test_cost_saving_non_negative(self) -> None:
        profile = [20.0, 50.0, 40.0, 10.0]
        result = simulate_peak_shaving(_battery(), profile, 30.0, 15.0)
        assert result.cost_saving >= 0.0

    def test_empty_profile_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_peak_shaving(_battery(), [], 30.0)

    def test_negative_demand_charge_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_peak_shaving(_battery(), [10.0, 20.0], 30.0, -5.0)

    @pytest.mark.parametrize("threshold", [20.0, 30.0, 40.0])
    def test_final_soc_in_range(self, threshold: float) -> None:
        profile = [10.0, 25.0, 45.0, 35.0, 15.0]
        result = simulate_peak_shaving(_battery(), profile, threshold)
        assert 0.0 <= result.final_soc_pct <= 100.0


class TestBatteryStorageEdgeCases:
    def test_battery_soc_never_exceeds_capacity(self) -> None:
        from app.battery_storage import Battery, simulate_peak_shaving

        b = Battery(
            capacity_kwh=50.0, max_charge_kw=20.0, max_discharge_kw=20.0, initial_soc_pct=100.0
        )
        profile = [5.0, 5.0, 5.0]
        result = simulate_peak_shaving(b, profile, 10.0)
        assert result.final_soc_pct <= 100.0

    def test_battery_soc_never_below_zero(self) -> None:
        from app.battery_storage import Battery, simulate_peak_shaving

        b = Battery(
            capacity_kwh=10.0, max_charge_kw=20.0, max_discharge_kw=20.0, initial_soc_pct=0.0
        )
        profile = [100.0, 100.0, 100.0]
        result = simulate_peak_shaving(b, profile, 5.0)
        assert result.final_soc_pct >= 0.0

    def test_battery_zero_capacity_raises(self) -> None:
        from app.battery_storage import Battery

        with pytest.raises(ValueError):
            Battery(capacity_kwh=0.0, max_charge_kw=10.0, max_discharge_kw=10.0)

    def test_peak_reduction_non_negative(self) -> None:
        from app.battery_storage import Battery, simulate_peak_shaving

        b = Battery(
            capacity_kwh=50.0, max_charge_kw=25.0, max_discharge_kw=25.0, initial_soc_pct=80.0
        )
        profile = [10.0, 60.0, 80.0, 40.0, 20.0]
        result = simulate_peak_shaving(b, profile, 50.0)
        assert result.peak_reduction_kw >= 0.0

    @pytest.mark.parametrize("soc_pct", [0.0, 50.0, 100.0])
    def test_valid_initial_soc(self, soc_pct: float) -> None:
        from app.battery_storage import Battery

        b = Battery(
            capacity_kwh=50.0, max_charge_kw=10.0, max_discharge_kw=10.0, initial_soc_pct=soc_pct
        )
        assert b.initial_soc_pct == soc_pct
