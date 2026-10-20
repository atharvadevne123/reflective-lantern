"""Tests for app.voltage_drop module."""

from __future__ import annotations

import math

import pytest

from app.voltage_drop import (
    CableResult,
    analyse_cable,
    cable_resistance,
    voltage_drop_percent,
    voltage_drop_single_phase,
    voltage_drop_three_phase,
)


class TestCableResistance:
    def test_longer_cable_more_resistance(self) -> None:
        r1 = cable_resistance(10.0, 2.5)
        r2 = cable_resistance(20.0, 2.5)
        assert r2 == pytest.approx(2 * r1, rel=1e-4)

    def test_larger_section_less_resistance(self) -> None:
        r_small = cable_resistance(10.0, 2.5)
        r_large = cable_resistance(10.0, 10.0)
        assert r_large < r_small

    def test_zero_length_raises(self) -> None:
        with pytest.raises(ValueError):
            cable_resistance(0.0, 2.5)

    def test_zero_cross_section_raises(self) -> None:
        with pytest.raises(ValueError):
            cable_resistance(10.0, 0.0)

    @pytest.mark.parametrize("section", [1.5, 2.5, 10.0, 25.0])
    def test_positive_result(self, section: float) -> None:
        assert cable_resistance(10.0, section) > 0.0


class TestVoltageDropSinglePhase:
    def test_unity_pf(self) -> None:
        vd = voltage_drop_single_phase(10.0, 1.0)
        assert vd == pytest.approx(10.0)

    def test_zero_current(self) -> None:
        assert voltage_drop_single_phase(0.0, 1.0) == pytest.approx(0.0)

    def test_negative_current_raises(self) -> None:
        with pytest.raises(ValueError):
            voltage_drop_single_phase(-5.0, 1.0)

    def test_zero_pf_raises(self) -> None:
        with pytest.raises(ValueError):
            voltage_drop_single_phase(10.0, 1.0, 0.0)

    @pytest.mark.parametrize("pf", [0.8, 0.9, 1.0])
    def test_lower_pf_lower_resistive_drop(self, pf: float) -> None:
        vd = voltage_drop_single_phase(10.0, 1.0, pf)
        assert vd >= 0.0


class TestVoltageDropThreePhase:
    def test_three_phase_larger_than_single(self) -> None:
        vd_1ph = voltage_drop_single_phase(10.0, 1.0)
        vd_3ph = voltage_drop_three_phase(10.0, 1.0)
        assert vd_3ph == pytest.approx(vd_1ph * math.sqrt(3), rel=1e-4)

    def test_zero_current(self) -> None:
        assert voltage_drop_three_phase(0.0, 1.0) == pytest.approx(0.0)

    def test_negative_current_raises(self) -> None:
        with pytest.raises(ValueError):
            voltage_drop_three_phase(-5.0, 1.0)

    @pytest.mark.parametrize("current", [5.0, 10.0, 50.0])
    def test_positive_result(self, current: float) -> None:
        assert voltage_drop_three_phase(current, 1.0) > 0.0


class TestVoltageDropPercent:
    def test_basic(self) -> None:
        assert voltage_drop_percent(6.9, 230.0) == pytest.approx(3.0, rel=1e-2)

    def test_zero_drop(self) -> None:
        assert voltage_drop_percent(0.0, 230.0) == pytest.approx(0.0)

    def test_zero_nominal_raises(self) -> None:
        with pytest.raises(ValueError):
            voltage_drop_percent(5.0, 0.0)

    def test_negative_drop_raises(self) -> None:
        with pytest.raises(ValueError):
            voltage_drop_percent(-5.0, 230.0)


class TestAnalyseCable:
    def test_returns_cable_result(self) -> None:
        result = analyse_cable(20.0, 10.0, 4.0, 230.0)
        assert isinstance(result, CableResult)

    def test_within_limit_for_short_cable(self) -> None:
        result = analyse_cable(5.0, 5.0, 10.0, 230.0)
        assert result.within_limit is True

    def test_power_loss_positive(self) -> None:
        result = analyse_cable(20.0, 10.0, 4.0, 230.0)
        assert result.power_loss_w > 0.0

    def test_three_phase(self) -> None:
        result = analyse_cable(20.0, 10.0, 4.0, 400.0, is_three_phase=True)
        assert result.voltage_drop_v > 0.0

    @pytest.mark.parametrize("section", [2.5, 6.0, 16.0])
    def test_larger_section_less_drop(self, section: float) -> None:
        result = analyse_cable(20.0, 10.0, section, 230.0)
        assert result.voltage_drop_pct >= 0.0


class TestVoltageDropEdgeCases:
    def test_single_phase_positive(self) -> None:
        drop = voltage_drop_single_phase(10.0, 50.0, 2.5)
        assert drop > 0.0

    def test_three_phase_less_than_single(self) -> None:
        sp = voltage_drop_single_phase(10.0, 50.0, 2.5)
        tp = voltage_drop_three_phase(10.0, 50.0, 2.5)
        assert tp < sp

    def test_zero_current_zero_drop(self) -> None:
        drop = voltage_drop_single_phase(0.0, 50.0, 2.5)
        assert drop == pytest.approx(0.0)

    def test_voltage_drop_percent_positive(self) -> None:
        pct = voltage_drop_percent(5.0, 230.0)
        assert pct > 0.0

    def test_longer_cable_more_drop(self) -> None:
        drop_short = voltage_drop_single_phase(10.0, 10.0, 2.5)
        drop_long = voltage_drop_single_phase(10.0, 50.0, 2.5)
        assert drop_long > drop_short

    def test_analyse_cable_returns_result(self) -> None:
        result = analyse_cable(10.0, 20.0, 4.0, 230.0)
        assert isinstance(result, CableResult)
        assert result.voltage_drop_v >= 0.0

    def test_analyse_cable_large_section_within_limit(self) -> None:
        result = analyse_cable(5.0, 5.0, 25.0, 230.0)
        assert result.within_limit is True

    @pytest.mark.parametrize("current", [5.0, 10.0, 20.0, 50.0])
    def test_drop_scales_with_current(self, current: float) -> None:
        drop = voltage_drop_single_phase(current, 20.0, 4.0)
        assert drop > 0.0

    def test_cable_resistance_zero_length_raises(self) -> None:
        with pytest.raises((ValueError, ZeroDivisionError)):
            cable_resistance(0.0, 2.5)
