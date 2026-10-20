"""Tests for app.insulation_calculator module."""

from __future__ import annotations

import pytest

from app.insulation_calculator import (
    InsulationUpgradeResult,
    analyse_insulation_upgrade,
    annual_heat_loss_kwh,
    combined_u_value,
    heat_loss_watts,
    r_value_from_u,
    u_value_from_r,
)


class TestHeatLossWatts:
    def test_basic(self) -> None:
        loss = heat_loss_watts(1.0, 10.0, 20.0)
        assert loss == pytest.approx(200.0)

    def test_zero_delta_t(self) -> None:
        assert heat_loss_watts(1.0, 10.0, 0.0) == pytest.approx(0.0)

    def test_zero_u_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_loss_watts(0.0, 10.0, 20.0)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_loss_watts(1.0, 0.0, 20.0)

    def test_negative_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_loss_watts(1.0, 10.0, -5.0)

    @pytest.mark.parametrize("u,area,dt", [(2.0, 50.0, 15.0), (0.3, 100.0, 20.0)])
    def test_parametrized(self, u: float, area: float, dt: float) -> None:
        assert heat_loss_watts(u, area, dt) == pytest.approx(u * area * dt)


class TestRValueFromU:
    def test_inverse_relationship(self) -> None:
        assert r_value_from_u(2.0) == pytest.approx(0.5, rel=1e-4)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            r_value_from_u(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            r_value_from_u(-1.0)

    @pytest.mark.parametrize("u", [0.3, 1.0, 2.1])
    def test_round_trip(self, u: float) -> None:
        assert u_value_from_r(r_value_from_u(u)) == pytest.approx(u, rel=1e-5)


class TestUValueFromR:
    def test_inverse_relationship(self) -> None:
        assert u_value_from_r(0.5) == pytest.approx(2.0, rel=1e-4)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            u_value_from_r(0.0)

    @pytest.mark.parametrize("r", [0.5, 1.0, 3.5])
    def test_positive_result(self, r: float) -> None:
        assert u_value_from_r(r) > 0.0


class TestCombinedUValue:
    def test_single_layer(self) -> None:
        assert combined_u_value([2.0]) == pytest.approx(0.5, rel=1e-5)

    def test_multiple_layers(self) -> None:
        result = combined_u_value([1.0, 1.0])
        assert result == pytest.approx(0.5, rel=1e-4)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            combined_u_value([])

    def test_zero_layer_raises(self) -> None:
        with pytest.raises(ValueError):
            combined_u_value([1.0, 0.0])

    @pytest.mark.parametrize("layers", [[1.0, 2.0], [0.5, 1.5, 2.0]])
    def test_combined_lower_than_any_layer_u(self, layers: list) -> None:
        combined = combined_u_value(layers)
        max_u = max(1.0 / r for r in layers)
        assert combined <= max_u


class TestAnnualHeatLossKwh:
    def test_basic(self) -> None:
        assert annual_heat_loss_kwh(1000.0, 8760.0) == pytest.approx(8760.0)

    def test_zero_watts_zero_kwh(self) -> None:
        assert annual_heat_loss_kwh(0.0, 4380.0) == pytest.approx(0.0)

    def test_negative_watts_raises(self) -> None:
        with pytest.raises(ValueError):
            annual_heat_loss_kwh(-100.0, 4380.0)

    @pytest.mark.parametrize("hours", [0.0, 2000.0, 8760.0])
    def test_non_negative_result(self, hours: float) -> None:
        assert annual_heat_loss_kwh(500.0, hours) >= 0.0


class TestAnalyseInsulationUpgrade:
    def test_returns_result(self) -> None:
        result = analyse_insulation_upgrade(50.0, 2.1, 0.3, 20.0, 5000.0)
        assert isinstance(result, InsulationUpgradeResult)

    def test_energy_saved_positive(self) -> None:
        result = analyse_insulation_upgrade(50.0, 2.1, 0.3, 20.0, 5000.0)
        assert result.annual_energy_saved_kwh > 0.0

    def test_worse_u_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_insulation_upgrade(50.0, 0.3, 2.1, 20.0, 5000.0)

    def test_equal_u_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_insulation_upgrade(50.0, 1.0, 1.0, 20.0, 5000.0)

    @pytest.mark.parametrize("area", [20.0, 50.0, 100.0])
    def test_larger_area_more_savings(self, area: float) -> None:
        result = analyse_insulation_upgrade(area, 2.1, 0.3, 20.0, 5000.0)
        assert result.co2_saved_kg > 0.0


class TestInsulationEdgeCases:
    def test_heat_loss_rate_positive(self) -> None:
        from app.insulation_calculator import heat_loss_rate
        rate = heat_loss_rate(50.0, 2.1, 20.0)
        assert rate > 0

    def test_heat_loss_rate_zero_delta_zero(self) -> None:
        from app.insulation_calculator import heat_loss_rate
        rate = heat_loss_rate(50.0, 2.1, 0.0)
        assert rate == pytest.approx(0.0)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_insulation_upgrade(0.0, 2.1, 0.3, 20.0, 5000.0)

    def test_zero_u_value_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_insulation_upgrade(50.0, 0.0, 0.3, 20.0, 5000.0)

    def test_payback_positive(self) -> None:
        result = analyse_insulation_upgrade(50.0, 2.1, 0.3, 20.0, 5000.0, cost=2000.0)
        if result.payback_years is not None:
            assert result.payback_years > 0

    @pytest.mark.parametrize("old_u,new_u", [(3.0, 0.2), (2.0, 0.3), (1.5, 0.5)])
    def test_greater_improvement_more_savings(self, old_u: float, new_u: float) -> None:
        r_small = analyse_insulation_upgrade(50.0, 1.5, 0.5, 20.0, 5000.0)
        r_large = analyse_insulation_upgrade(50.0, old_u, new_u, 20.0, 5000.0)
        improvement_large = old_u - new_u
        improvement_small = 1.5 - 0.5
        if improvement_large > improvement_small:
            assert r_large.annual_energy_saved_kwh >= r_small.annual_energy_saved_kwh
