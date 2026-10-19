"""Tests for app.hvac_estimator module."""

from __future__ import annotations

import pytest

from app.hvac_estimator import (
    HVACEstimate,
    cooling_degree_days,
    cooling_kwh_from_cdd,
    estimate_hvac,
    heating_degree_days,
    heating_kwh_from_hdd,
)


class TestHeatingDegreeDays:
    def test_all_cold_days(self) -> None:
        hdd = heating_degree_days([10.0, 12.0, 8.0], base_temp=18.0)
        assert hdd == pytest.approx(18.0 - 10.0 + 18.0 - 12.0 + 18.0 - 8.0)

    def test_warm_weather_zero_hdd(self) -> None:
        assert heating_degree_days([25.0, 30.0], base_temp=18.0) == 0.0

    def test_mixed_temps(self) -> None:
        hdd = heating_degree_days([20.0, 10.0], base_temp=18.0)
        assert hdd == pytest.approx(8.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            heating_degree_days([])

    @pytest.mark.parametrize(
        "temps,base,expected",
        [
            ([18.0, 18.0], 18.0, 0.0),
            ([0.0, 0.0], 18.0, 36.0),
            ([10.0], 15.0, 5.0),
        ],
    )
    def test_parametrized(
        self, temps: list[float], base: float, expected: float
    ) -> None:
        assert heating_degree_days(temps, base) == pytest.approx(expected)


class TestCoolingDegreeDays:
    def test_all_hot_days(self) -> None:
        cdd = cooling_degree_days([25.0, 30.0], base_temp=18.0)
        assert cdd == pytest.approx(7.0 + 12.0)

    def test_cold_weather_zero_cdd(self) -> None:
        assert cooling_degree_days([5.0, 10.0], base_temp=18.0) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            cooling_degree_days([])

    @pytest.mark.parametrize(
        "temps,base,expected",
        [
            ([18.0], 18.0, 0.0),
            ([20.0, 22.0], 18.0, 6.0),
        ],
    )
    def test_parametrized(
        self, temps: list[float], base: float, expected: float
    ) -> None:
        assert cooling_degree_days(temps, base) == pytest.approx(expected)


class TestHeatingKwhFromHDD:
    def test_zero_hdd_zero_heating(self) -> None:
        assert heating_kwh_from_hdd(0.0, 1.0) == 0.0

    def test_basic_calculation(self) -> None:
        result = heating_kwh_from_hdd(100.0, 0.1, hours_per_day=24.0, efficiency=1.0)
        assert result == pytest.approx(240.0, rel=1e-4)

    def test_efficiency_increases_consumption(self) -> None:
        high_eff = heating_kwh_from_hdd(100.0, 0.1, efficiency=0.9)
        low_eff = heating_kwh_from_hdd(100.0, 0.1, efficiency=0.7)
        assert low_eff > high_eff

    def test_negative_hdd_raises(self) -> None:
        with pytest.raises(ValueError):
            heating_kwh_from_hdd(-10.0, 0.1)

    def test_invalid_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            heating_kwh_from_hdd(100.0, 0.1, efficiency=0.0)

    @pytest.mark.parametrize(
        "hdd,hlc,eff",
        [(50.0, 0.2, 0.9), (200.0, 0.1, 0.85), (0.0, 0.5, 0.95)],
    )
    def test_returns_non_negative(
        self, hdd: float, hlc: float, eff: float
    ) -> None:
        assert heating_kwh_from_hdd(hdd, hlc, efficiency=eff) >= 0.0


class TestCoolingKwhFromCDD:
    def test_zero_cdd_zero_cooling(self) -> None:
        assert cooling_kwh_from_cdd(0.0, 1.0) == 0.0

    def test_basic_calculation(self) -> None:
        result = cooling_kwh_from_cdd(100.0, 0.1, cop=2.0, hours_per_day=24.0)
        assert result == pytest.approx(120.0, rel=1e-4)

    def test_higher_cop_lower_consumption(self) -> None:
        low_cop = cooling_kwh_from_cdd(100.0, 0.1, cop=2.0)
        high_cop = cooling_kwh_from_cdd(100.0, 0.1, cop=4.0)
        assert high_cop < low_cop

    def test_zero_cop_raises(self) -> None:
        with pytest.raises(ValueError):
            cooling_kwh_from_cdd(100.0, 0.1, cop=0.0)

    def test_negative_cdd_raises(self) -> None:
        with pytest.raises(ValueError):
            cooling_kwh_from_cdd(-5.0, 0.1)


class TestEstimateHVAC:
    def _temps(self) -> list[float]:
        return [10.0] * 90 + [25.0] * 90

    def test_returns_hvac_estimate(self) -> None:
        result = estimate_hvac(self._temps(), 0.1, 0.05)
        assert isinstance(result, HVACEstimate)

    def test_total_is_sum_of_parts(self) -> None:
        result = estimate_hvac(self._temps(), 0.1, 0.05)
        assert result.total_kwh == pytest.approx(
            result.heating_kwh + result.cooling_kwh, rel=1e-5
        )

    def test_cost_proportional_to_rate(self) -> None:
        r1 = estimate_hvac(self._temps(), 0.1, 0.05, energy_rate=0.20)
        r2 = estimate_hvac(self._temps(), 0.1, 0.05, energy_rate=0.40)
        assert r2.estimated_cost == pytest.approx(2 * r1.estimated_cost, rel=1e-5)

    def test_hdd_cdd_non_negative(self) -> None:
        result = estimate_hvac(self._temps(), 0.1, 0.05)
        assert result.hdd >= 0.0
        assert result.cdd >= 0.0

    @pytest.mark.parametrize(
        "hlc,shg",
        [(0.05, 0.03), (0.2, 0.1), (0.5, 0.2)],
    )
    def test_parametrized_loss_coefficients(self, hlc: float, shg: float) -> None:
        result = estimate_hvac(self._temps(), hlc, shg)
        assert result.total_kwh >= 0.0
