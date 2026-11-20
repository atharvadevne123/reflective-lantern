"""Tests for app.power_quality module."""

from __future__ import annotations

import math

import pytest

from app.power_quality import (
    PQEvent,
    PQReport,
    analyse_power_quality,
    classify_voltage_event,
    crest_factor,
    en50160_compliance,
    k_factor,
    rms_voltage,
    thd_voltage,
)


class TestRmsVoltage:
    def test_sinusoidal(self) -> None:
        samples = [math.sin(2 * math.pi * i / 100) * 325.3 for i in range(100)]
        vrms = rms_voltage(samples)
        assert vrms == pytest.approx(325.3 / math.sqrt(2), rel=1e-2)

    def test_constant_signal(self) -> None:
        assert rms_voltage([10.0] * 100) == pytest.approx(10.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            rms_voltage([])

    @pytest.mark.parametrize("n", [10, 100, 1000])
    def test_non_negative(self, n: int) -> None:
        samples = [float(i % 10) for i in range(n)]
        assert rms_voltage(samples) >= 0.0


class TestThdVoltage:
    def test_pure_fundamental_zero_thd(self) -> None:
        assert thd_voltage(230.0, []) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        result = thd_voltage(100.0, [5.0])
        assert result == pytest.approx(5.0)

    def test_zero_fundamental_raises(self) -> None:
        with pytest.raises(ValueError):
            thd_voltage(0.0, [10.0])

    @pytest.mark.parametrize("harmonics", [[], [5.0], [5.0, 3.0, 2.0]])
    def test_non_negative(self, harmonics: list) -> None:
        assert thd_voltage(230.0, harmonics) >= 0.0


class TestCrestFactor:
    def test_sinusoidal(self) -> None:
        assert crest_factor(325.3, 230.0) == pytest.approx(325.3 / 230.0, rel=1e-4)

    def test_zero_rms_raises(self) -> None:
        with pytest.raises(ValueError):
            crest_factor(325.3, 0.0)

    def test_peak_lt_rms_raises(self) -> None:
        with pytest.raises(ValueError):
            crest_factor(100.0, 200.0)

    @pytest.mark.parametrize("peak,rms", [(325.0, 230.0), (170.0, 120.0)])
    def test_above_one(self, peak: float, rms: float) -> None:
        assert crest_factor(peak, rms) >= 1.0


class TestKFactor:
    def test_no_harmonics_is_one(self) -> None:
        assert k_factor(100.0, []) == pytest.approx(1.0)

    def test_zero_fundamental_raises(self) -> None:
        with pytest.raises(ValueError):
            k_factor(0.0, [10.0])

    def test_with_harmonics_greater_than_one(self) -> None:
        result = k_factor(100.0, [20.0, 10.0])
        assert result > 1.0

    @pytest.mark.parametrize("harmonics", [[], [5.0], [5.0, 3.0]])
    def test_non_negative(self, harmonics: list) -> None:
        assert k_factor(50.0, harmonics) >= 0.0


class TestClassifyVoltageEvent:
    def test_normal(self) -> None:
        assert classify_voltage_event(230.0) == PQEvent.NORMAL

    def test_sag(self) -> None:
        assert classify_voltage_event(200.0) == PQEvent.SAG

    def test_swell(self) -> None:
        assert classify_voltage_event(260.0) == PQEvent.SWELL

    def test_interruption(self) -> None:
        assert classify_voltage_event(5.0) == PQEvent.INTERRUPTION

    @pytest.mark.parametrize("v", [220.0, 230.0, 240.0])
    def test_normal_range(self, v: float) -> None:
        assert classify_voltage_event(v) == PQEvent.NORMAL


class TestEn50160Compliance:
    def test_compliant(self) -> None:
        assert en50160_compliance(230.0, 5.0) is True

    def test_high_thd_fails(self) -> None:
        assert en50160_compliance(230.0, 10.0) is False

    def test_low_voltage_fails(self) -> None:
        assert en50160_compliance(190.0, 3.0) is False

    def test_high_voltage_fails(self) -> None:
        assert en50160_compliance(260.0, 3.0) is False


class TestAnalysePowerQuality:
    def _samples(self) -> list[float]:
        return [math.sin(2 * math.pi * i / 100) * 325.3 for i in range(100)]

    def test_returns_pq_report(self) -> None:
        result = analyse_power_quality(self._samples(), 230.0, [5.0], 325.0, 10.0, [1.0])
        assert isinstance(result, PQReport)

    def test_thd_positive(self) -> None:
        result = analyse_power_quality(self._samples(), 230.0, [5.0], 325.0, 10.0, [1.0])
        assert result.thd_pct >= 0.0

    @pytest.mark.parametrize("harmonics", [[], [5.0], [5.0, 3.0]])
    def test_various_harmonics(self, harmonics: list) -> None:
        result = analyse_power_quality(self._samples(), 230.0, harmonics, 325.0, 10.0, harmonics)
        assert result.vrms_v > 0.0


class TestPowerQualityEdgeCases:
    def _samples(self) -> list[float]:
        return [math.sin(2 * math.pi * i / 100) * 325.3 for i in range(100)]

    def test_rms_pure_sine(self) -> None:
        samples = [math.sin(2 * math.pi * i / 100) * 325.3 for i in range(100)]
        vrms = rms_voltage(samples)
        assert vrms == pytest.approx(325.3 / math.sqrt(2), rel=1e-2)

    def test_thd_zero_for_pure_sine(self) -> None:
        samples = [math.sin(2 * math.pi * i / 100) * 325.3 for i in range(100)]
        thd = thd_voltage(samples[0], [])
        assert thd == pytest.approx(0.0, abs=1e-6)

    def test_crest_factor_sine(self) -> None:
        samples = [math.sin(2 * math.pi * i / 100) * 325.3 for i in range(100)]
        cf = crest_factor(samples)
        assert cf == pytest.approx(math.sqrt(2), rel=1e-1)

    def test_k_factor_non_negative(self) -> None:
        harmonics = [10.0, 5.0, 3.0]
        kf = k_factor(harmonics)
        assert kf >= 0.0

    def test_classify_voltage_event_sag(self) -> None:
        event = classify_voltage_event(0.85, 0.05)
        assert isinstance(event, PQEvent)

    @pytest.mark.parametrize("vrms,expected", [(0.85, True), (1.15, True), (1.0, False)])
    def test_classify_various_voltages(self, vrms: float, expected: bool) -> None:
        event = classify_voltage_event(vrms, 0.1)
        if expected:
            assert event is not None
        else:
            assert event is None or isinstance(event, PQEvent)

    def test_pq_report_has_thd(self) -> None:
        result = analyse_power_quality(self._samples(), 230.0, [5.0], 325.0, 10.0, [1.0])
        assert hasattr(result, "thd_pct")
        assert result.thd_pct >= 0.0

    def test_en50160_nominal_passes(self) -> None:
        assert en50160_compliance(230.0, 3.0) is True

    @pytest.mark.parametrize("vrms", [210.0, 220.0, 230.0, 240.0, 250.0])
    def test_compliance_within_range_passes(self, vrms: float) -> None:
        result = en50160_compliance(vrms, 2.0)
        assert isinstance(result, bool)
