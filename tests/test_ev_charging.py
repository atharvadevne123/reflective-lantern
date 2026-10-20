"""Tests for app.ev_charging module."""

from __future__ import annotations

import pytest

from app.ev_charging import (
    ChargerLevel,
    ChargingResult,
    ChargingSession,
    charging_duration_hours,
    energy_to_charge,
    fleet_daily_demand_kw,
    simulate_session,
    smart_charge_schedule,
)


class TestEnergyToCharge:
    def test_full_charge(self) -> None:
        kwh = energy_to_charge(0.0, 100.0, 70.0, 1.0)
        assert kwh == pytest.approx(70.0)

    def test_with_efficiency(self) -> None:
        kwh = energy_to_charge(0.0, 100.0, 60.0, 0.9)
        assert kwh == pytest.approx(60.0 / 0.9, rel=1e-4)

    def test_partial_charge(self) -> None:
        kwh = energy_to_charge(50.0, 80.0, 100.0, 1.0)
        assert kwh == pytest.approx(30.0)

    def test_invalid_soc_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_charge(-10.0, 100.0, 60.0)

    def test_start_ge_target_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_charge(80.0, 80.0, 60.0)

    def test_zero_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_charge(0.0, 100.0, 60.0, 0.0)

    @pytest.mark.parametrize(
        "start,target,batt,eff",
        [(0.0, 50.0, 100.0, 0.9), (20.0, 80.0, 60.0, 0.95)],
    )
    def test_positive_result(
        self, start: float, target: float, batt: float, eff: float
    ) -> None:
        assert energy_to_charge(start, target, batt, eff) > 0.0


class TestChargingDurationHours:
    def test_level_2_duration(self) -> None:
        duration = charging_duration_hours(36.0, ChargerLevel.LEVEL_2)
        assert duration == pytest.approx(5.0, rel=1e-4)

    def test_dc_fast_shorter_than_level2(self) -> None:
        l2 = charging_duration_hours(40.0, ChargerLevel.LEVEL_2)
        dc = charging_duration_hours(40.0, ChargerLevel.DC_FAST)
        assert dc < l2

    def test_zero_energy_zero_duration(self) -> None:
        assert charging_duration_hours(0.0, ChargerLevel.LEVEL_1) == 0.0

    def test_negative_energy_raises(self) -> None:
        with pytest.raises(ValueError):
            charging_duration_hours(-10.0, ChargerLevel.LEVEL_2)

    @pytest.mark.parametrize("level", list(ChargerLevel))
    def test_all_levels_positive_for_positive_energy(self, level: ChargerLevel) -> None:
        assert charging_duration_hours(20.0, level) > 0.0


class TestSimulateSession:
    def _session(self) -> ChargingSession:
        return ChargingSession(
            charger_level=ChargerLevel.LEVEL_2,
            start_soc_pct=20.0,
            target_soc_pct=80.0,
            battery_kwh=60.0,
        )

    def test_returns_charging_result(self) -> None:
        result = simulate_session(self._session())
        assert isinstance(result, ChargingResult)

    def test_energy_added_lt_drawn(self) -> None:
        result = simulate_session(self._session())
        assert result.energy_added_kwh <= result.energy_drawn_kwh

    def test_cost_positive(self) -> None:
        result = simulate_session(self._session(), energy_rate=0.28)
        assert result.cost > 0.0

    @pytest.mark.parametrize("rate", [0.15, 0.28, 0.50])
    def test_parametrized_rates(self, rate: float) -> None:
        result = simulate_session(self._session(), energy_rate=rate)
        assert result.cost > 0.0


class TestFleetDailyDemandKw:
    def test_single_vehicle(self) -> None:
        demand = fleet_daily_demand_kw(1, 30.0, 4.0, 0.9, 8.0)
        assert demand > 0.0

    def test_more_vehicles_more_demand(self) -> None:
        d1 = fleet_daily_demand_kw(5, 30.0, 4.0, 0.9, 8.0)
        d2 = fleet_daily_demand_kw(10, 30.0, 4.0, 0.9, 8.0)
        assert d2 == pytest.approx(2 * d1, rel=1e-5)

    def test_zero_vehicles_raises(self) -> None:
        with pytest.raises(ValueError):
            fleet_daily_demand_kw(0, 30.0, 4.0, 0.9, 8.0)

    def test_invalid_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            fleet_daily_demand_kw(5, 30.0, 4.0, 0.0, 8.0)

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_parametrized_fleet_sizes(self, n: int) -> None:
        assert fleet_daily_demand_kw(n, 25.0, 3.5, 0.9, 8.0) > 0.0


class TestSmartChargeSchedule:
    def test_returns_correct_length(self) -> None:
        schedule = smart_charge_schedule(30.0, 6, [0.10, 0.15, 0.30, 0.08, 0.20, 0.12])
        assert len(schedule) == 6

    def test_total_kwh_allocated(self) -> None:
        schedule = smart_charge_schedule(30.0, 6, [0.10, 0.15, 0.30, 0.08, 0.20, 0.12])
        assert sum(schedule) == pytest.approx(30.0, rel=1e-4)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            smart_charge_schedule(30.0, 6, [0.10, 0.20])

    def test_negative_total_raises(self) -> None:
        with pytest.raises(ValueError):
            smart_charge_schedule(-10.0, 4, [0.1, 0.2, 0.3, 0.4])

    @pytest.mark.parametrize("total", [10.0, 30.0, 60.0])
    def test_parametrized_totals(self, total: float) -> None:
        rates = [0.10, 0.15, 0.30, 0.08]
        schedule = smart_charge_schedule(total, 4, rates)
        assert sum(schedule) == pytest.approx(total, rel=1e-4)


class TestEVChargingEdgeCases:
    def test_energy_to_charge_full_state(self) -> None:
        energy = energy_to_charge(60.0, 1.0, 1.0)
        assert energy == pytest.approx(0.0, abs=1e-6)

    def test_energy_to_charge_invalid_soc_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_charge(60.0, 1.1, 1.0)

    def test_energy_to_charge_target_below_initial_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_charge(60.0, 0.8, 0.5)

    def test_charging_duration_zero_energy_zero_duration(self) -> None:
        duration = charging_duration_hours(0.0, 7.4)
        assert duration == pytest.approx(0.0)

    def test_simulate_session_returns_result(self) -> None:
        session = ChargingSession(
            battery_capacity_kwh=60.0,
            initial_soc=0.2,
            target_soc=0.9,
            charger_level=ChargerLevel.LEVEL_2,
        )
        result = simulate_session(session)
        assert isinstance(result, ChargingResult)
        assert result.energy_kwh > 0

    def test_fleet_demand_scales_with_fleet_size(self) -> None:
        d1 = fleet_daily_demand_kw(1, 60.0, 0.2, 0.9)
        d5 = fleet_daily_demand_kw(5, 60.0, 0.2, 0.9)
        assert d5 == pytest.approx(d1 * 5, rel=1e-3)

    @pytest.mark.parametrize("charger", list(ChargerLevel))
    def test_all_charger_levels(self, charger: ChargerLevel) -> None:
        session = ChargingSession(60.0, 0.2, 0.8, charger)
        result = simulate_session(session)
        assert result.duration_hours > 0
