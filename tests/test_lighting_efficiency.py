"""Tests for app.lighting_efficiency module."""

from __future__ import annotations

import pytest

from app.lighting_efficiency import (
    LightingAuditResult,
    annual_kwh,
    audit_space,
    luminous_efficacy,
    power_for_target_lux,
    upgrade_savings,
)


class TestLuminousEfficacy:
    def test_known_value(self) -> None:
        assert luminous_efficacy(1300.0, 10.0) == pytest.approx(130.0)

    def test_zero_lumens(self) -> None:
        assert luminous_efficacy(0.0, 10.0) == pytest.approx(0.0)

    def test_zero_watts_raises(self) -> None:
        with pytest.raises(ValueError):
            luminous_efficacy(1000.0, 0.0)

    def test_negative_lumens_raises(self) -> None:
        with pytest.raises(ValueError):
            luminous_efficacy(-100.0, 10.0)

    @pytest.mark.parametrize(
        "lumens,watts,expected",
        [(800.0, 10.0, 80.0), (1300.0, 10.0, 130.0), (600.0, 60.0, 10.0)],
    )
    def test_parametrized(self, lumens: float, watts: float, expected: float) -> None:
        assert luminous_efficacy(lumens, watts) == pytest.approx(expected)


class TestPowerForTargetLux:
    def test_basic_calculation(self) -> None:
        watts = power_for_target_lux(
            area_m2=50.0, target_lux=500.0, maintenance_factor=0.8, utilisation_factor=0.7
        )
        assert watts > 0.0

    def test_higher_efficacy_lower_power(self) -> None:
        low = power_for_target_lux(50.0, 500.0, 0.8, 0.7, 60.0)
        high = power_for_target_lux(50.0, 500.0, 0.8, 0.7, 130.0)
        assert high < low

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            power_for_target_lux(0.0, 500.0, 0.8, 0.7)

    def test_invalid_maintenance_factor_raises(self) -> None:
        with pytest.raises(ValueError):
            power_for_target_lux(50.0, 500.0, 0.0, 0.7)

    @pytest.mark.parametrize(
        "area,lux",
        [(20.0, 300.0), (100.0, 500.0), (200.0, 750.0)],
    )
    def test_positive_result(self, area: float, lux: float) -> None:
        assert power_for_target_lux(area, lux, 0.8, 0.7) > 0.0


class TestAnnualKwh:
    def test_simple_calculation(self) -> None:
        result = annual_kwh(1000.0, 10.0, days_per_year=365)
        assert result == pytest.approx(3650.0)

    def test_zero_hours_zero_kwh(self) -> None:
        assert annual_kwh(1000.0, 0.0) == pytest.approx(0.0)

    def test_occupancy_reduces_consumption(self) -> None:
        full = annual_kwh(1000.0, 10.0, 1.0)
        half = annual_kwh(1000.0, 10.0, 0.5)
        assert half == pytest.approx(full * 0.5)

    def test_negative_watts_raises(self) -> None:
        with pytest.raises(ValueError):
            annual_kwh(-100.0, 10.0)

    def test_invalid_occupancy_raises(self) -> None:
        with pytest.raises(ValueError):
            annual_kwh(100.0, 10.0, 1.5)

    @pytest.mark.parametrize("hours", [0.0, 8.0, 12.0, 24.0])
    def test_non_negative_result(self, hours: float) -> None:
        assert annual_kwh(500.0, hours) >= 0.0


class TestUpgradeSavings:
    def test_basic_savings(self) -> None:
        saved_kwh, payback = upgrade_savings(1000.0, 300.0, 10.0, 0.28, 5000.0)
        assert saved_kwh > 0.0
        assert payback > 0.0

    def test_new_equals_old_raises(self) -> None:
        with pytest.raises(ValueError):
            upgrade_savings(500.0, 500.0, 8.0, 0.28, 2000.0)

    def test_new_exceeds_old_raises(self) -> None:
        with pytest.raises(ValueError):
            upgrade_savings(300.0, 500.0, 8.0, 0.28, 2000.0)

    def test_negative_args_raise(self) -> None:
        with pytest.raises(ValueError):
            upgrade_savings(-100.0, 50.0, 8.0, 0.28, 1000.0)

    @pytest.mark.parametrize(
        "old,new,hours",
        [(1000.0, 300.0, 10.0), (500.0, 100.0, 8.0)],
    )
    def test_savings_positive(self, old: float, new: float, hours: float) -> None:
        saved, _ = upgrade_savings(old, new, hours, 0.28, 3000.0)
        assert saved > 0.0


class TestAuditSpace:
    def test_returns_audit_result(self) -> None:
        result = audit_space(100.0, 2000.0, 10.0)
        assert isinstance(result, LightingAuditResult)

    def test_annual_kwh_positive(self) -> None:
        result = audit_space(100.0, 2000.0, 8.0)
        assert result.annual_kwh > 0.0

    def test_annual_cost_positive(self) -> None:
        result = audit_space(100.0, 2000.0, 8.0)
        assert result.annual_cost > 0.0

    @pytest.mark.parametrize("area", [50.0, 100.0, 500.0])
    def test_parametrized_areas(self, area: float) -> None:
        result = audit_space(area, area * 10, 10.0)
        assert result.annual_kwh > 0.0


class TestLightingEdgeCases:
    def test_luminous_efficacy_higher_lumens_higher_efficacy(self) -> None:
        e1 = luminous_efficacy(1000.0, 10.0)
        e2 = luminous_efficacy(2000.0, 10.0)
        assert e2 > e1

    def test_annual_kwh_formula(self) -> None:
        kwh = annual_kwh(100.0, 8.0)
        assert kwh == pytest.approx(100.0 * 8.0 * 365.0 / 1000.0, rel=1e-4)

    def test_annual_kwh_zero_power_zero(self) -> None:
        assert annual_kwh(0.0, 8.0) == pytest.approx(0.0)

    def test_power_for_target_lux_positive(self) -> None:
        power = power_for_target_lux(100.0, 500.0, 0.8, 0.65)
        assert power > 0.0

    def test_upgrade_savings_positive(self) -> None:
        savings_kwh, payback = upgrade_savings(100.0, 50.0, 8.0, 0.28, 500.0)
        assert savings_kwh > 0.0

    def test_upgrade_savings_equal_watts_raises(self) -> None:
        with pytest.raises(ValueError):
            upgrade_savings(100.0, 100.0, 8.0, 0.28, 0.0)

    def test_audit_space_returns_result(self) -> None:
        result = audit_space(200.0, 4000.0, 10.0)
        assert isinstance(result, LightingAuditResult)
        assert result.installed_power_w > 0

    @pytest.mark.parametrize("hours", [1.0, 4.0, 8.0, 12.0])
    def test_annual_kwh_scales_with_hours(self, hours: float) -> None:
        kwh = annual_kwh(100.0, hours)
        assert kwh > 0.0
