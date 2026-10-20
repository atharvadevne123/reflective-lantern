"""Tests for app.tidal_power module."""

from __future__ import annotations

import pytest

from app.tidal_power import (
    BarrageResult,
    TidalStreamResult,
    current_power_density,
    minimum_cut_in_speed,
    tidal_barrage_energy,
    tidal_stream_power,
)


class TestTidalStreamPower:
    def test_basic_output(self) -> None:
        result = tidal_stream_power(10.0, 2.5)
        assert isinstance(result, TidalStreamResult)
        assert result.actual_power_kw > 0

    def test_zero_current_gives_zero_power(self) -> None:
        result = tidal_stream_power(10.0, 0.0)
        assert result.actual_power_kw == 0.0

    def test_zero_diameter_raises(self) -> None:
        with pytest.raises(ValueError):
            tidal_stream_power(0.0, 2.5)

    def test_negative_diameter_raises(self) -> None:
        with pytest.raises(ValueError):
            tidal_stream_power(-5.0, 2.5)

    def test_negative_speed_raises(self) -> None:
        with pytest.raises(ValueError):
            tidal_stream_power(10.0, -1.0)

    def test_cp_above_betz_raises(self) -> None:
        with pytest.raises(ValueError):
            tidal_stream_power(10.0, 2.5, power_coefficient=0.60)

    def test_power_scales_with_velocity_cubed(self) -> None:
        r1 = tidal_stream_power(10.0, 2.0)
        r2 = tidal_stream_power(10.0, 4.0)
        assert r2.actual_power_kw == pytest.approx(r1.actual_power_kw * 8.0, rel=1e-3)

    def test_rotor_area_computed(self) -> None:
        import math
        result = tidal_stream_power(10.0, 2.5)
        expected_area = math.pi * 25.0
        assert result.rotor_area_m2 == pytest.approx(expected_area, rel=1e-3)

    @pytest.mark.parametrize("diameter,speed", [(5.0, 1.5), (15.0, 3.0), (20.0, 2.0)])
    def test_positive_annual_mwh(self, diameter: float, speed: float) -> None:
        result = tidal_stream_power(diameter, speed)
        assert result.annual_mwh >= 0

    def test_co2_saved_positive(self) -> None:
        result = tidal_stream_power(10.0, 2.5)
        assert result.co2_saved_kg > 0


class TestTidalBarrageEnergy:
    def test_basic(self) -> None:
        result = tidal_barrage_energy(1e6, 6.0)
        assert isinstance(result, BarrageResult)
        assert result.electrical_energy_kwh > 0

    def test_zero_basin_raises(self) -> None:
        with pytest.raises(ValueError):
            tidal_barrage_energy(0.0, 6.0)

    def test_zero_range_raises(self) -> None:
        with pytest.raises(ValueError):
            tidal_barrage_energy(1e6, 0.0)

    def test_invalid_turbine_eff_raises(self) -> None:
        with pytest.raises(ValueError):
            tidal_barrage_energy(1e6, 6.0, turbine_efficiency=0.0)

    def test_invalid_generator_eff_raises(self) -> None:
        with pytest.raises(ValueError):
            tidal_barrage_energy(1e6, 6.0, generator_efficiency=1.1)

    def test_electrical_less_than_potential(self) -> None:
        result = tidal_barrage_energy(1e6, 6.0)
        assert result.electrical_energy_kwh < result.potential_energy_kwh

    def test_more_cycles_more_energy(self) -> None:
        r2 = tidal_barrage_energy(1e6, 6.0, tidal_cycles_per_day=2.0)
        r4 = tidal_barrage_energy(1e6, 6.0, tidal_cycles_per_day=4.0)
        assert r4.electrical_energy_kwh > r2.electrical_energy_kwh


class TestCurrentPowerDensity:
    def test_zero_speed(self) -> None:
        assert current_power_density(0.0) == 0.0

    def test_known_value(self) -> None:
        density = current_power_density(2.0)
        expected = 0.5 * 1025.0 * 8.0
        assert density == pytest.approx(expected, rel=1e-3)

    def test_negative_speed_raises(self) -> None:
        with pytest.raises(ValueError):
            current_power_density(-1.0)

    @pytest.mark.parametrize("speed", [1.0, 2.0, 3.0])
    def test_cubic_scaling(self, speed: float) -> None:
        d1 = current_power_density(1.0)
        d = current_power_density(speed)
        assert d == pytest.approx(d1 * speed**3, rel=1e-3)


class TestMinimumCutInSpeed:
    def test_basic(self) -> None:
        speed = minimum_cut_in_speed(500.0, 15.0)
        assert speed > 0

    def test_zero_power_raises(self) -> None:
        with pytest.raises(ValueError):
            minimum_cut_in_speed(0.0, 15.0)

    def test_zero_diameter_raises(self) -> None:
        with pytest.raises(ValueError):
            minimum_cut_in_speed(500.0, 0.0)

    def test_larger_rotor_needs_lower_speed(self) -> None:
        s1 = minimum_cut_in_speed(500.0, 10.0)
        s2 = minimum_cut_in_speed(500.0, 20.0)
        assert s2 < s1
