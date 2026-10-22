"""Tests for app.distributed_energy module."""

from __future__ import annotations

import pytest

from app.distributed_energy import (
    DERPortfolio,
    DERStatus,
    DERType,
    DERUnit,
    dispatch_order,
    grid_import_export,
    self_consumption_rate,
    self_sufficiency_rate,
)


class TestDERUnit:
    def _make_unit(self, output: float = 50.0, status: DERStatus = DERStatus.ONLINE) -> DERUnit:
        u = DERUnit("u1", DERType.SOLAR_PV, 100.0, output, status)
        return u

    def test_utilisation_pct(self) -> None:
        u = self._make_unit(output=75.0)
        assert u.utilisation_pct == pytest.approx(75.0)

    def test_available_kw_online(self) -> None:
        u = self._make_unit(output=60.0, status=DERStatus.ONLINE)
        assert u.available_kw == pytest.approx(40.0)

    def test_available_kw_fault_is_zero(self) -> None:
        u = self._make_unit(status=DERStatus.FAULT)
        assert u.available_kw == 0.0

    def test_zero_rated_power_raises(self) -> None:
        with pytest.raises(ValueError):
            DERUnit("u1", DERType.BATTERY, 0.0)

    def test_negative_output_raises(self) -> None:
        with pytest.raises(ValueError):
            DERUnit("u1", DERType.WIND, 100.0, current_output_kw=-5.0)

    def test_priority_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            DERUnit("u1", DERType.SOLAR_PV, 100.0, priority=0)


class TestDERPortfolio:
    def _make_portfolio(self) -> DERPortfolio:
        assets = [
            DERUnit("pv1", DERType.SOLAR_PV, 100.0, 80.0, DERStatus.ONLINE),
            DERUnit("bat1", DERType.BATTERY, 50.0, 20.0, DERStatus.ONLINE),
            DERUnit("wind1", DERType.WIND, 200.0, 0.0, DERStatus.OFFLINE),
        ]
        return DERPortfolio(assets)

    def test_total_rated_kw(self) -> None:
        p = self._make_portfolio()
        assert p.total_rated_kw() == pytest.approx(350.0)

    def test_total_online_kw(self) -> None:
        p = self._make_portfolio()
        assert p.total_online_kw() == pytest.approx(100.0)

    def test_assets_by_type(self) -> None:
        p = self._make_portfolio()
        solar = p.assets_by_type(DERType.SOLAR_PV)
        assert len(solar) == 1

    def test_assets_online(self) -> None:
        p = self._make_portfolio()
        online = p.assets_online()
        assert len(online) == 2

    def test_utilisation_pct(self) -> None:
        p = self._make_portfolio()
        assert 0 <= p.utilisation_pct() <= 100

    def test_available_kw_excludes_offline(self) -> None:
        p = self._make_portfolio()
        avail = p.available_kw()
        assert avail >= 0


class TestDispatchOrder:
    def _make_portfolio(self) -> DERPortfolio:
        assets = [
            DERUnit("pv1", DERType.SOLAR_PV, 100.0, 50.0, DERStatus.ONLINE, priority=3),
            DERUnit("bat1", DERType.BATTERY, 50.0, 10.0, DERStatus.STANDBY, priority=1),
            DERUnit("chp1", DERType.CHP, 200.0, 100.0, DERStatus.ONLINE, priority=5),
        ]
        return DERPortfolio(assets)

    def test_returns_list(self) -> None:
        result = dispatch_order(self._make_portfolio(), 50.0)
        assert isinstance(result, list)

    def test_total_dispatched_covers_demand(self) -> None:
        p = self._make_portfolio()
        result = dispatch_order(p, 30.0)
        total = sum(kw for _, kw in result)
        assert total >= 30.0 or total == p.available_kw()

    def test_zero_demand_empty_dispatch(self) -> None:
        result = dispatch_order(self._make_portfolio(), 0.0)
        assert result == []

    def test_negative_demand_raises(self) -> None:
        with pytest.raises(ValueError):
            dispatch_order(self._make_portfolio(), -10.0)


class TestGridImportExport:
    def test_import(self) -> None:
        assert grid_import_export(100.0, 60.0) == pytest.approx(40.0)

    def test_export(self) -> None:
        assert grid_import_export(50.0, 80.0) == pytest.approx(-30.0)

    def test_balanced(self) -> None:
        assert grid_import_export(100.0, 100.0) == pytest.approx(0.0)

    def test_negative_load_raises(self) -> None:
        with pytest.raises(ValueError):
            grid_import_export(-10.0, 50.0)

    def test_negative_generation_raises(self) -> None:
        with pytest.raises(ValueError):
            grid_import_export(100.0, -10.0)


class TestSelfConsumptionRate:
    def test_all_consumed(self) -> None:
        assert self_consumption_rate(50.0, 100.0) == pytest.approx(1.0)

    def test_partial_consumption(self) -> None:
        assert self_consumption_rate(100.0, 60.0) == pytest.approx(0.6)

    def test_zero_generation_returns_zero(self) -> None:
        assert self_consumption_rate(0.0, 100.0) == 0.0

    def test_negative_generation_raises(self) -> None:
        with pytest.raises(ValueError):
            self_consumption_rate(-10.0, 100.0)


class TestSelfSufficiencyRate:
    def test_fully_sufficient(self) -> None:
        assert self_sufficiency_rate(200.0, 100.0) == pytest.approx(1.0)

    def test_partial_sufficiency(self) -> None:
        assert self_sufficiency_rate(60.0, 100.0) == pytest.approx(0.6)

    def test_zero_load_returns_one(self) -> None:
        assert self_sufficiency_rate(50.0, 0.0) == pytest.approx(1.0)

    def test_negative_load_raises(self) -> None:
        with pytest.raises(ValueError):
            self_sufficiency_rate(100.0, -10.0)

    @pytest.mark.parametrize(
        "gen,load,expected",
        [
            (100.0, 200.0, 0.5),
            (150.0, 100.0, 1.0),
            (0.0, 100.0, 0.0),
        ],
    )
    def test_parametrized(self, gen: float, load: float, expected: float) -> None:
        assert self_sufficiency_rate(gen, load) == pytest.approx(expected)
