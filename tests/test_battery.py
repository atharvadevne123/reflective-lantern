"""Tests for app/battery.py."""

from __future__ import annotations

import pytest

from app.battery import (
    BatterySpec,
    demand_charge_saving,
    peak_shave,
    required_capacity_kwh,
)

SPIKY_LOAD = [10.0, 10.0, 50.0, 60.0, 55.0, 10.0, 10.0]
FLAT_LOAD = [20.0] * 8


def make_spec(**overrides: float) -> BatterySpec:
    """Build a battery spec with generous defaults, overridable per test."""
    params: dict[str, float] = {
        "capacity_kwh": 200.0,
        "max_charge_kw": 100.0,
        "max_discharge_kw": 100.0,
    }
    params.update(overrides)
    return BatterySpec(**params)  # type: ignore[arg-type]


class TestBatterySpec:
    def test_usable_capacity_applies_depth_of_discharge(self) -> None:
        spec = BatterySpec(capacity_kwh=100.0, max_charge_kw=10.0, max_discharge_kw=10.0, max_depth_of_discharge=0.8)
        assert spec.usable_kwh == pytest.approx(80.0)

    def test_full_depth_of_discharge_uses_whole_pack(self) -> None:
        spec = BatterySpec(capacity_kwh=100.0, max_charge_kw=10.0, max_discharge_kw=10.0, max_depth_of_discharge=1.0)
        assert spec.usable_kwh == pytest.approx(100.0)

    @pytest.mark.parametrize("capacity", [0.0, -10.0])
    def test_non_positive_capacity_rejected(self, capacity: float) -> None:
        with pytest.raises(ValueError, match="capacity_kwh must be positive"):
            BatterySpec(capacity_kwh=capacity, max_charge_kw=10.0, max_discharge_kw=10.0)

    @pytest.mark.parametrize(("charge", "discharge"), [(0.0, 10.0), (10.0, 0.0), (-5.0, 10.0)])
    def test_non_positive_power_rejected(self, charge: float, discharge: float) -> None:
        with pytest.raises(ValueError, match="power limits must be positive"):
            BatterySpec(capacity_kwh=100.0, max_charge_kw=charge, max_discharge_kw=discharge)

    @pytest.mark.parametrize("efficiency", [0.0, -0.5, 1.5])
    def test_invalid_efficiency_rejected(self, efficiency: float) -> None:
        with pytest.raises(ValueError, match=r"round_trip_efficiency must be in \(0, 1\]"):
            BatterySpec(capacity_kwh=100.0, max_charge_kw=10.0, max_discharge_kw=10.0, round_trip_efficiency=efficiency)

    @pytest.mark.parametrize("dod", [0.0, -0.5, 1.5])
    def test_invalid_depth_of_discharge_rejected(self, dod: float) -> None:
        with pytest.raises(ValueError, match=r"max_depth_of_discharge must be in \(0, 1\]"):
            BatterySpec(capacity_kwh=100.0, max_charge_kw=10.0, max_discharge_kw=10.0, max_depth_of_discharge=dod)


class TestPeakShave:
    def test_ample_battery_defends_the_target(self) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=30.0)
        assert result.peak_after_kw <= 30.0
        assert result.peak_reduction_kw > 0

    def test_discharge_power_limit_caps_reduction(self) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(max_discharge_kw=25.0), target_peak_kw=30.0)
        # The 60 kW hour can only be cut by 25 kW.
        assert result.peak_after_kw == pytest.approx(35.0)

    def test_undersized_pack_cannot_defend_target(self, caplog: pytest.LogCaptureFixture) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(capacity_kwh=5.0), target_peak_kw=30.0)
        assert result.peak_after_kw > 30.0
        assert "could not defend" in caplog.text

    def test_flat_load_below_target_needs_no_discharge(self) -> None:
        result = peak_shave(FLAT_LOAD, make_spec(), target_peak_kw=50.0)
        assert result.energy_discharged_kwh == 0.0
        assert result.peak_reduction_kw <= 0.0 or result.peak_before_kw == result.peak_after_kw

    def test_grid_series_matches_input_length(self) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=30.0)
        assert len(result.grid_hourly_kw) == len(SPIKY_LOAD)

    def test_charging_draws_more_than_it_stores(self) -> None:
        # The pack starts full, so it must discharge into the spike first;
        # refilling that same energy afterwards costs more than it returned,
        # because round-trip losses land on the way in.
        result = peak_shave([40.0, 5.0, 5.0, 5.0], make_spec(round_trip_efficiency=0.5), target_peak_kw=20.0)
        assert result.energy_discharged_kwh == pytest.approx(20.0)
        assert result.energy_charged_kwh > result.energy_discharged_kwh

    def test_starts_charged_so_early_spike_is_covered(self) -> None:
        # No opportunity to charge before the spike, yet it is still shaved.
        result = peak_shave([40.0, 5.0], make_spec(), target_peak_kw=20.0)
        assert result.peak_after_kw <= 20.0

    def test_empty_load_returns_zeroed_result(self) -> None:
        result = peak_shave([], make_spec(), target_peak_kw=30.0)
        assert result.peak_before_kw == 0.0
        assert result.grid_hourly_kw == []

    def test_reduction_pct_matches_absolute_reduction(self) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=30.0)
        expected = round(100.0 * result.peak_reduction_kw / result.peak_before_kw, 2)
        assert result.peak_reduction_pct == pytest.approx(expected)

    def test_cycles_track_discharged_energy(self) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=30.0)
        spec = make_spec()
        assert result.equivalent_cycles == pytest.approx(result.energy_discharged_kwh / spec.usable_kwh, rel=1e-3)

    def test_degradation_grows_with_cycling(self) -> None:
        light = peak_shave(FLAT_LOAD, make_spec(), target_peak_kw=50.0)
        heavy = peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=30.0)
        assert heavy.capacity_lost_pct >= light.capacity_lost_pct

    def test_negative_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_peak_kw must be non-negative"):
            peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=-1.0)

    def test_negative_load_rejected(self) -> None:
        with pytest.raises(ValueError, match="load must be non-negative"):
            peak_shave([10.0, -5.0], make_spec(), target_peak_kw=30.0)


class TestRequiredCapacityKwh:
    def test_sizes_for_the_largest_excursion(self) -> None:
        # Excursion above 30: 20 + 30 + 25 = 75 kWh.
        assert required_capacity_kwh(SPIKY_LOAD, 30.0) == pytest.approx(75.0)

    def test_load_under_target_needs_nothing(self) -> None:
        assert required_capacity_kwh(FLAT_LOAD, 50.0) == 0.0

    def test_resets_between_separated_excursions(self) -> None:
        # Two 10 kWh excursions separated by a dip: size for one, not both.
        assert required_capacity_kwh([20.0, 5.0, 20.0], 10.0) == pytest.approx(10.0)

    def test_lower_target_needs_more_capacity(self) -> None:
        assert required_capacity_kwh(SPIKY_LOAD, 20.0) > required_capacity_kwh(SPIKY_LOAD, 40.0)

    def test_empty_load_needs_nothing(self) -> None:
        assert required_capacity_kwh([], 30.0) == 0.0

    def test_negative_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_peak_kw must be non-negative"):
            required_capacity_kwh(SPIKY_LOAD, -1.0)

    def test_sizing_guidance_lets_battery_hold_target(self) -> None:
        needed = required_capacity_kwh(SPIKY_LOAD, 30.0)
        spec = BatterySpec(
            capacity_kwh=needed,
            max_charge_kw=100.0,
            max_discharge_kw=100.0,
            max_depth_of_discharge=1.0,
        )
        assert peak_shave(SPIKY_LOAD, spec, target_peak_kw=30.0).peak_after_kw <= 30.0


class TestDemandChargeSaving:
    def test_scales_with_peak_reduction(self) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=30.0)
        assert demand_charge_saving(result, 15.0) == pytest.approx(round(result.peak_reduction_kw * 15.0, 2))

    def test_zero_tariff_saves_nothing(self) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=30.0)
        assert demand_charge_saving(result, 0.0) == 0.0

    def test_no_reduction_saves_nothing(self) -> None:
        result = peak_shave(FLAT_LOAD, make_spec(), target_peak_kw=50.0)
        assert demand_charge_saving(result, 15.0) <= 0.0

    def test_negative_tariff_rejected(self) -> None:
        result = peak_shave(SPIKY_LOAD, make_spec(), target_peak_kw=30.0)
        with pytest.raises(ValueError, match="demand_charge_per_kw must be non-negative"):
            demand_charge_saving(result, -1.0)
