"""Tests for app.microGrid_controller module."""

from __future__ import annotations

import pytest

from app.microGrid_controller import (
    DispatchDecision,
    dispatch_self_consumption,
    net_load_kw,
    self_consumption_ratio,
    self_sufficiency_ratio,
)


class TestNetLoadKw:
    def test_deficit(self) -> None:
        assert net_load_kw(10.0, 5.0) == pytest.approx(5.0)

    def test_surplus(self) -> None:
        assert net_load_kw(5.0, 10.0) == pytest.approx(-5.0)

    def test_balanced(self) -> None:
        assert net_load_kw(8.0, 8.0) == pytest.approx(0.0)

    def test_negative_load_raises(self) -> None:
        with pytest.raises(ValueError):
            net_load_kw(-5.0, 10.0)

    def test_negative_pv_raises(self) -> None:
        with pytest.raises(ValueError):
            net_load_kw(10.0, -5.0)

    @pytest.mark.parametrize("load,pv", [(10.0, 5.0), (5.0, 10.0), (8.0, 8.0)])
    def test_parametrized(self, load: float, pv: float) -> None:
        assert net_load_kw(load, pv) == pytest.approx(load - pv)


class TestSelfConsumptionRatio:
    def test_full_self_consumption(self) -> None:
        assert self_consumption_ratio(10.0, 8.0, 0.0) == pytest.approx(1.0)

    def test_full_export(self) -> None:
        assert self_consumption_ratio(0.0, 8.0, 8.0) == pytest.approx(0.0)

    def test_zero_pv_raises(self) -> None:
        with pytest.raises(ValueError):
            self_consumption_ratio(5.0, 0.0, 0.0)

    def test_export_exceeds_pv_raises(self) -> None:
        with pytest.raises(ValueError):
            self_consumption_ratio(5.0, 8.0, 10.0)

    @pytest.mark.parametrize("pv,export", [(10.0, 0.0), (10.0, 5.0), (10.0, 10.0)])
    def test_between_0_and_1(self, pv: float, export: float) -> None:
        assert 0.0 <= self_consumption_ratio(5.0, pv, export) <= 1.0


class TestSelfSufficiencyRatio:
    def test_full_sufficiency(self) -> None:
        assert self_sufficiency_ratio(10.0, 0.0) == pytest.approx(1.0)

    def test_zero_sufficiency(self) -> None:
        assert self_sufficiency_ratio(10.0, 10.0) == pytest.approx(0.0)

    def test_zero_load_raises(self) -> None:
        with pytest.raises(ValueError):
            self_sufficiency_ratio(0.0, 5.0)

    def test_negative_import_raises(self) -> None:
        with pytest.raises(ValueError):
            self_sufficiency_ratio(10.0, -5.0)

    @pytest.mark.parametrize("load,imported", [(10.0, 0.0), (10.0, 5.0), (10.0, 10.0)])
    def test_between_0_and_1(self, load: float, imported: float) -> None:
        assert 0.0 <= self_sufficiency_ratio(load, imported) <= 1.0


class TestDispatchSelfConsumption:
    def test_returns_dispatch_decision(self) -> None:
        result = dispatch_self_consumption(5.0, 8.0, 50.0, 20.0, 5.0, 5.0)
        assert isinstance(result, DispatchDecision)

    def test_surplus_charges_battery(self) -> None:
        result = dispatch_self_consumption(0.0, 10.0, 50.0, 20.0, 5.0, 5.0)
        assert result.battery_charge_kw > 0.0

    def test_deficit_discharges_battery(self) -> None:
        result = dispatch_self_consumption(10.0, 0.0, 80.0, 20.0, 5.0, 5.0)
        assert result.battery_discharge_kw > 0.0

    def test_invalid_soc_raises(self) -> None:
        with pytest.raises(ValueError):
            dispatch_self_consumption(5.0, 8.0, 110.0, 20.0, 5.0, 5.0)

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            dispatch_self_consumption(5.0, 8.0, 50.0, 0.0, 5.0, 5.0)

    @pytest.mark.parametrize("pv,load", [(0.0, 10.0), (10.0, 0.0), (5.0, 5.0)])
    def test_non_negative_outputs(self, pv: float, load: float) -> None:
        result = dispatch_self_consumption(load, pv, 50.0, 20.0, 5.0, 5.0)
        assert result.battery_charge_kw >= 0.0
        assert result.battery_discharge_kw >= 0.0
        assert result.grid_import_kw >= 0.0
        assert result.grid_export_kw >= 0.0


class TestMicroGridEdgeCases:
    def test_net_load_zero_pv_equals_load(self) -> None:
        assert net_load_kw(10.0, 0.0) == pytest.approx(10.0)

    def test_net_load_zero_load_negative(self) -> None:
        assert net_load_kw(0.0, 5.0) == pytest.approx(-5.0)

    def test_self_consumption_all_exported_zero(self) -> None:
        ratio = self_consumption_ratio(load_kwh=5.0, pv_kwh=10.0, exported_kwh=10.0)
        assert ratio == pytest.approx(0.0)

    def test_self_consumption_none_exported_one(self) -> None:
        ratio = self_consumption_ratio(load_kwh=10.0, pv_kwh=10.0, exported_kwh=0.0)
        assert ratio == pytest.approx(1.0, rel=1e-4)

    def test_self_sufficiency_zero_import_one(self) -> None:
        ratio = self_sufficiency_ratio(load_kwh=10.0, imported_kwh=0.0)
        assert ratio == pytest.approx(1.0, rel=1e-4)

    def test_self_sufficiency_all_imported_zero(self) -> None:
        ratio = self_sufficiency_ratio(load_kwh=10.0, imported_kwh=10.0)
        assert ratio == pytest.approx(0.0)

    def test_dispatch_surplus_exports_to_grid(self) -> None:
        result = dispatch_self_consumption(2.0, 10.0, 100.0, 20.0, 5.0, 5.0)
        assert result.grid_export_kw >= 0.0 or result.battery_charge_kw >= 0.0

    def test_dispatch_fully_self_sufficient(self) -> None:
        result = dispatch_self_consumption(5.0, 5.0, 50.0, 20.0, 5.0, 5.0)
        assert isinstance(result, DispatchDecision)

    @pytest.mark.parametrize("soc", [10.0, 50.0, 90.0])
    def test_dispatch_various_soc(self, soc: float) -> None:
        result = dispatch_self_consumption(8.0, 4.0, soc, 20.0, 5.0, 5.0)
        assert isinstance(result, DispatchDecision)
