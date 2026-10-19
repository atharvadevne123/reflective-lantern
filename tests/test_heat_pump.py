"""Tests for app.heat_pump module."""

from __future__ import annotations

import pytest

from app.heat_pump import (
    HeatPumpResult,
    analyse_heat_pump,
    carnot_cop,
    electricity_for_heat,
    heat_delivered_kwh,
    realistic_cop,
    seasonal_cop,
)


class TestCarnotCop:
    def test_known_value(self) -> None:
        cop = carnot_cop(0.0, 50.0)
        assert cop == pytest.approx(323.15 / 50.0, rel=1e-4)

    def test_sink_le_source_raises(self) -> None:
        with pytest.raises(ValueError):
            carnot_cop(40.0, 30.0)

    def test_equal_temps_raises(self) -> None:
        with pytest.raises(ValueError):
            carnot_cop(20.0, 20.0)

    @pytest.mark.parametrize("source,sink", [(-10.0, 45.0), (0.0, 55.0), (10.0, 35.0)])
    def test_cop_above_one(self, source: float, sink: float) -> None:
        assert carnot_cop(source, sink) > 1.0


class TestRealisticCop:
    def test_less_than_carnot(self) -> None:
        carnot = carnot_cop(0.0, 50.0)
        real = realistic_cop(0.0, 50.0, 0.5)
        assert real < carnot

    def test_zero_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            realistic_cop(0.0, 50.0, 0.0)

    def test_invalid_temps_raises(self) -> None:
        with pytest.raises(ValueError):
            realistic_cop(50.0, 40.0)

    @pytest.mark.parametrize("ef", [0.3, 0.5, 0.8])
    def test_higher_efficiency_higher_cop(self, ef: float) -> None:
        cop = realistic_cop(0.0, 50.0, ef)
        assert cop > 0.0


class TestHeatDeliveredKwh:
    def test_basic(self) -> None:
        assert heat_delivered_kwh(100.0, 3.0) == pytest.approx(300.0)

    def test_zero_electricity(self) -> None:
        assert heat_delivered_kwh(0.0, 3.0) == pytest.approx(0.0)

    def test_negative_electricity_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_delivered_kwh(-10.0, 3.0)

    def test_zero_cop_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_delivered_kwh(100.0, 0.0)

    @pytest.mark.parametrize("cop", [2.0, 3.5, 5.0])
    def test_higher_cop_more_heat(self, cop: float) -> None:
        assert heat_delivered_kwh(100.0, cop) == pytest.approx(100.0 * cop)


class TestElectricityForHeat:
    def test_basic(self) -> None:
        assert electricity_for_heat(300.0, 3.0) == pytest.approx(100.0)

    def test_zero_heat(self) -> None:
        assert electricity_for_heat(0.0, 3.0) == pytest.approx(0.0)

    def test_negative_heat_raises(self) -> None:
        with pytest.raises(ValueError):
            electricity_for_heat(-100.0, 3.0)

    def test_zero_cop_raises(self) -> None:
        with pytest.raises(ValueError):
            electricity_for_heat(300.0, 0.0)

    @pytest.mark.parametrize("heat,cop", [(1000.0, 2.5), (5000.0, 4.0)])
    def test_parametrized(self, heat: float, cop: float) -> None:
        assert electricity_for_heat(heat, cop) == pytest.approx(heat / cop)


class TestSeasonalCop:
    def test_uniform_cops(self) -> None:
        cops = [3.0] * 12
        heat = [1000.0] * 12
        assert seasonal_cop(cops, heat) == pytest.approx(3.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            seasonal_cop([3.0] * 12, [1000.0] * 6)

    def test_zero_cop_raises(self) -> None:
        cops = [0.0] + [3.0] * 11
        with pytest.raises(ValueError):
            seasonal_cop(cops, [1000.0] * 12)

    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_positive_result(self, n: int) -> None:
        cops = [3.0] * n
        heat = [500.0] * n
        assert seasonal_cop(cops, heat) > 0.0


class TestAnalyseHeatPump:
    def test_returns_result(self) -> None:
        result = analyse_heat_pump(5.0, 45.0, 10000.0)
        assert isinstance(result, HeatPumpResult)

    def test_cop_above_one(self) -> None:
        result = analyse_heat_pump(5.0, 45.0, 10000.0)
        assert result.cop > 1.0

    def test_elec_less_than_heat(self) -> None:
        result = analyse_heat_pump(5.0, 45.0, 10000.0)
        assert result.electricity_consumed_kwh < result.heat_delivered_kwh

    @pytest.mark.parametrize("source", [-5.0, 5.0, 10.0])
    def test_warmer_source_better_cop(self, source: float) -> None:
        result = analyse_heat_pump(source, 45.0, 10000.0)
        assert result.cop > 1.0
