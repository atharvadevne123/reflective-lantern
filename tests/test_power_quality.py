"""Tests for app/power_quality.py."""

from __future__ import annotations

import math

import pytest

from app.power_quality import (
    GOOD_POWER_FACTOR,
    MAX_VOLTAGE_IMBALANCE_PCT,
    apparent_power,
    build_report,
    correction_kvar,
    power_factor,
    rate_power_factor,
    reactive_power,
    voltage_imbalance,
)

BALANCED_PHASES = [230.0, 230.0, 230.0]
SKEWED_PHASES = [230.0, 245.0, 215.0]


class TestPowerFactor:
    def test_unity_when_all_power_is_real(self) -> None:
        assert power_factor(100.0, 100.0) == pytest.approx(1.0)

    def test_partial_load(self) -> None:
        assert power_factor(80.0, 100.0) == pytest.approx(0.8)

    def test_zero_apparent_power_returns_zero(self) -> None:
        assert power_factor(0.0, 0.0) == 0.0

    def test_real_exceeding_apparent_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed apparent power"):
            power_factor(150.0, 100.0)

    @pytest.mark.parametrize(("real", "apparent"), [(-1.0, 100.0), (100.0, -1.0)])
    def test_negative_power_rejected(self, real: float, apparent: float) -> None:
        with pytest.raises(ValueError, match="power must be non-negative"):
            power_factor(real, apparent)


class TestApparentPower:
    def test_pythagorean_combination(self) -> None:
        assert apparent_power(3.0, 4.0) == pytest.approx(5.0)

    def test_no_reactive_component_equals_real(self) -> None:
        assert apparent_power(100.0, 0.0) == pytest.approx(100.0)

    def test_never_below_real_power(self) -> None:
        assert apparent_power(100.0, 50.0) >= 100.0

    def test_negative_reactive_treated_by_magnitude(self) -> None:
        assert apparent_power(3.0, -4.0) == pytest.approx(5.0)

    def test_negative_real_rejected(self) -> None:
        with pytest.raises(ValueError, match="real_power_kw must be non-negative"):
            apparent_power(-1.0, 4.0)


class TestReactivePower:
    def test_unity_power_factor_has_no_reactive(self) -> None:
        assert reactive_power(100.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_lower_power_factor_needs_more_reactive(self) -> None:
        assert reactive_power(100.0, 0.7) > reactive_power(100.0, 0.9)

    def test_round_trips_through_apparent_power(self) -> None:
        kvar = reactive_power(100.0, 0.8)
        assert apparent_power(100.0, kvar) == pytest.approx(125.0, rel=1e-3)

    def test_matches_closed_form(self) -> None:
        expected = 100.0 * math.tan(math.acos(0.85))
        assert reactive_power(100.0, 0.85) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("pf", [0.0, -0.5, 1.5])
    def test_out_of_range_power_factor_rejected(self, pf: float) -> None:
        with pytest.raises(ValueError, match=r"power_factor_value must be in \(0, 1\]"):
            reactive_power(100.0, pf)


class TestRatePowerFactor:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1.0, "good"), (GOOD_POWER_FACTOR, "good"), (0.90, "acceptable"), (0.85, "acceptable"), (0.70, "poor")],
    )
    def test_buckets(self, value: float, expected: str) -> None:
        assert rate_power_factor(value) == expected


class TestCorrectionKvar:
    def test_already_at_target_needs_nothing(self) -> None:
        assert correction_kvar(100.0, 0.98) == 0.0

    def test_poor_factor_needs_correction(self) -> None:
        assert correction_kvar(100.0, 0.75) > 0

    def test_worse_factor_needs_more_correction(self) -> None:
        assert correction_kvar(100.0, 0.70) > correction_kvar(100.0, 0.90)

    def test_correction_reaches_target(self) -> None:
        real, current = 100.0, 0.80
        kvar_now = reactive_power(real, current)
        kvar_after = kvar_now - correction_kvar(real, current, target_power_factor=0.95)
        assert power_factor(real, apparent_power(real, kvar_after)) == pytest.approx(0.95, abs=1e-3)

    def test_invalid_target_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"target_power_factor must be in \(0, 1\]"):
            correction_kvar(100.0, 0.8, target_power_factor=1.5)


class TestVoltageImbalance:
    def test_balanced_phases_have_none(self) -> None:
        assert voltage_imbalance(BALANCED_PHASES) == pytest.approx(0.0)

    def test_skewed_phases_report_imbalance(self) -> None:
        assert voltage_imbalance(SKEWED_PHASES) > 0

    def test_wider_spread_raises_imbalance(self) -> None:
        assert voltage_imbalance([230.0, 260.0, 200.0]) > voltage_imbalance(SKEWED_PHASES)

    def test_zero_voltage_returns_zero(self) -> None:
        assert voltage_imbalance([0.0, 0.0]) == 0.0

    def test_two_phases_accepted(self) -> None:
        assert voltage_imbalance([230.0, 220.0]) > 0

    def test_single_phase_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2 phase voltages"):
            voltage_imbalance([230.0])

    def test_negative_voltage_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            voltage_imbalance([230.0, -10.0])


class TestBuildReport:
    def test_healthy_site_passes_both_checks(self) -> None:
        report = build_report(100.0, 10.0, BALANCED_PHASES)
        assert report.power_factor_rating == "good"
        assert report.imbalance_within_limit is True

    def test_poor_factor_is_flagged(self, caplog: pytest.LogCaptureFixture) -> None:
        report = build_report(100.0, 90.0, BALANCED_PHASES)
        assert report.power_factor_rating == "poor"
        assert "Poor power factor" in caplog.text

    def test_excess_imbalance_is_flagged(self, caplog: pytest.LogCaptureFixture) -> None:
        report = build_report(100.0, 10.0, [230.0, 280.0, 190.0])
        assert report.imbalance_within_limit is False
        assert report.voltage_imbalance_pct > MAX_VOLTAGE_IMBALANCE_PCT
        assert "Voltage imbalance" in caplog.text

    def test_apparent_power_is_consistent(self) -> None:
        report = build_report(100.0, 40.0, BALANCED_PHASES)
        assert report.apparent_power_kva == pytest.approx(apparent_power(100.0, 40.0))
        assert report.power_factor == pytest.approx(100.0 / report.apparent_power_kva, rel=1e-3)

    def test_rating_matches_standalone_classifier(self) -> None:
        report = build_report(100.0, 55.0, BALANCED_PHASES)
        assert report.power_factor_rating == rate_power_factor(report.power_factor)

    def test_single_phase_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2 phase voltages"):
            build_report(100.0, 10.0, [230.0])
