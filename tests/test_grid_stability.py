"""Tests for app.grid_stability module."""

from __future__ import annotations

import math

import pytest

from app.grid_stability import (
    apparent_power_kva,
    grid_metrics,
    is_frequency_normal,
    is_voltage_normal,
    power_factor,
    reactive_power_from_pf,
    thd_percent,
    voltage_unbalance_percent,
)


class TestPowerFactor:
    def test_unity_pf(self) -> None:
        assert power_factor(10.0, 0.0) == pytest.approx(1.0)

    def test_zero_apparent_returns_zero(self) -> None:
        assert power_factor(0.0, 0.0) == 0.0

    def test_lagging_pf(self) -> None:
        pf = power_factor(3.0, 4.0)
        assert pf == pytest.approx(0.6, rel=1e-4)

    def test_negative_active_raises(self) -> None:
        with pytest.raises(ValueError):
            power_factor(-1.0, 0.0)

    @pytest.mark.parametrize(
        "active,reactive,expected_pf",
        [
            (4.0, 3.0, 0.8),
            (10.0, 0.0, 1.0),
            (0.0, 5.0, 0.0),
        ],
    )
    def test_parametrized(
        self, active: float, reactive: float, expected_pf: float
    ) -> None:
        assert power_factor(active, reactive) == pytest.approx(expected_pf, rel=1e-4)


class TestApparentPower:
    def test_known_value(self) -> None:
        assert apparent_power_kva(3.0, 4.0) == pytest.approx(5.0, rel=1e-4)

    def test_zero_inputs(self) -> None:
        assert apparent_power_kva(0.0, 0.0) == 0.0

    @pytest.mark.parametrize(
        "active,reactive,expected",
        [(6.0, 8.0, 10.0), (5.0, 12.0, 13.0), (8.0, 15.0, 17.0)],
    )
    def test_pythagorean_triples(
        self, active: float, reactive: float, expected: float
    ) -> None:
        assert apparent_power_kva(active, reactive) == pytest.approx(expected, rel=1e-4)


class TestReactivePowerFromPF:
    def test_unity_pf_gives_zero_reactive(self) -> None:
        assert reactive_power_from_pf(10.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_half_pf(self) -> None:
        pf = 0.5
        q = reactive_power_from_pf(10.0, pf)
        angle = math.acos(pf)
        assert q == pytest.approx(10.0 * math.tan(angle), rel=1e-4)

    def test_invalid_pf_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            reactive_power_from_pf(10.0, 0.0)

    def test_negative_active_raises(self) -> None:
        with pytest.raises(ValueError):
            reactive_power_from_pf(-5.0, 0.9)


class TestVoltageUnbalance:
    def test_balanced_three_phase(self) -> None:
        assert voltage_unbalance_percent(230.0, 230.0, 230.0) == pytest.approx(0.0)

    def test_unbalanced(self) -> None:
        result = voltage_unbalance_percent(220.0, 230.0, 240.0)
        assert result > 0.0

    def test_all_zero(self) -> None:
        assert voltage_unbalance_percent(0.0, 0.0, 0.0) == 0.0

    def test_negative_voltage_raises(self) -> None:
        with pytest.raises(ValueError):
            voltage_unbalance_percent(-1.0, 230.0, 230.0)

    @pytest.mark.parametrize(
        "va,vb,vc",
        [(230.0, 230.0, 230.0), (228.0, 230.0, 232.0), (225.0, 230.0, 235.0)],
    )
    def test_returns_non_negative(self, va: float, vb: float, vc: float) -> None:
        assert voltage_unbalance_percent(va, vb, vc) >= 0.0


class TestVoltageNormal:
    def test_nominal_is_normal(self) -> None:
        assert is_voltage_normal(230.0) is True

    def test_low_voltage_abnormal(self) -> None:
        assert is_voltage_normal(200.0) is False

    def test_high_voltage_abnormal(self) -> None:
        assert is_voltage_normal(260.0) is False

    @pytest.mark.parametrize("v", [207.0, 230.0, 253.0])
    def test_within_tolerance(self, v: float) -> None:
        assert is_voltage_normal(v) is True


class TestFrequencyNormal:
    def test_nominal_is_normal(self) -> None:
        assert is_frequency_normal(50.0) is True

    def test_out_of_range(self) -> None:
        assert is_frequency_normal(49.0) is False

    @pytest.mark.parametrize("hz", [49.5, 50.0, 50.5])
    def test_within_tolerance(self, hz: float) -> None:
        assert is_frequency_normal(hz) is True


class TestTHD:
    def test_zero_harmonics(self) -> None:
        assert thd_percent([], 100.0) == 0.0

    def test_known_thd(self) -> None:
        # harmonics = [30], fundamental = 100 → THD = 30%
        assert thd_percent([30.0], 100.0) == pytest.approx(30.0, rel=1e-4)

    def test_zero_fundamental_raises(self) -> None:
        with pytest.raises(ValueError):
            thd_percent([10.0], 0.0)

    def test_negative_harmonic_raises(self) -> None:
        with pytest.raises(ValueError):
            thd_percent([-5.0], 100.0)


class TestGridMetrics:
    def test_returns_dataclass(self) -> None:
        m = grid_metrics(230.0, 50.0, 5.0, 3.0)
        assert m.active_power_kw == 5.0
        assert m.apparent_power_kva == pytest.approx(
            apparent_power_kva(5.0, 3.0), rel=1e-4
        )

    def test_voltage_normal_flag(self) -> None:
        m = grid_metrics(230.0, 50.0, 10.0, 0.0)
        assert m.is_voltage_normal is True

    def test_frequency_normal_flag(self) -> None:
        m = grid_metrics(230.0, 50.0, 10.0, 0.0)
        assert m.is_frequency_normal is True

    def test_with_harmonics(self) -> None:
        m = grid_metrics(230.0, 50.0, 5.0, 3.0, harmonics=[30.0])
        assert m.thd_percent > 0.0

    def test_with_phase_voltages(self) -> None:
        m = grid_metrics(
            230.0, 50.0, 5.0, 3.0, phase_voltages=(228.0, 230.0, 232.0)
        )
        assert m.voltage_unbalance_percent >= 0.0


class TestGridStabilityEdgeCases:
    def test_power_factor_exactly_half(self) -> None:
        pf = power_factor(3.0, 3.0 * (3**0.5))
        assert pf == pytest.approx(0.5, rel=1e-3)

    def test_thd_all_zero_harmonics(self) -> None:
        assert thd_percent([], 100.0) == 0.0

    def test_thd_multiple_harmonics(self) -> None:
        result = thd_percent([30.0, 20.0, 10.0], 100.0)
        import math
        expected = 100.0 * math.sqrt(30**2 + 20**2 + 10**2) / 100.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_voltage_above_tolerance_not_normal(self) -> None:
        assert is_voltage_normal(260.0) is False

    def test_voltage_below_tolerance_not_normal(self) -> None:
        assert is_voltage_normal(200.0) is False

    def test_frequency_50hz_normal(self) -> None:
        assert is_frequency_normal(50.0) is True

    def test_frequency_50_6hz_abnormal(self) -> None:
        assert is_frequency_normal(50.6) is False

    @pytest.mark.parametrize("voltage,expected", [
        (230.0, True),
        (245.0, True),
        (215.0, True),
        (210.0, False),
        (255.0, False),
    ])
    def test_voltage_range_parametrized(self, voltage: float, expected: bool) -> None:
        assert is_voltage_normal(voltage) is expected

    def test_unbalance_symmetric_zero(self) -> None:
        assert voltage_unbalance_percent(240.0, 240.0, 240.0) == pytest.approx(0.0)

    def test_reactive_power_zero_at_unity_pf(self) -> None:
        q = reactive_power_from_pf(100.0, 1.0)
        assert q == pytest.approx(0.0, abs=1e-6)

    def test_grid_metrics_zero_loads_ok(self) -> None:
        m = grid_metrics(230.0, 50.0, 0.0, 0.0)
        assert m.power_factor == 0.0
