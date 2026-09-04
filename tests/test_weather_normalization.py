"""Tests for app/weather_normalization.py."""

from __future__ import annotations

import pytest

from app.weather_normalization import (
    DEFAULT_BASE_TEMPERATURE_C,
    compare_periods,
    cooling_degree_days,
    heating_degree_days,
    normalization_factor,
    normalize_consumption,
)

COLD_WEEK = [2.0, 4.0, 1.0, 0.0, 3.0, 5.0, 2.0]
HOT_WEEK = [28.0, 30.0, 32.0, 29.0, 31.0, 33.0, 30.0]
MILD_WEEK = [DEFAULT_BASE_TEMPERATURE_C] * 7


class TestHeatingDegreeDays:
    def test_cold_week_accumulates(self) -> None:
        expected = sum(DEFAULT_BASE_TEMPERATURE_C - t for t in COLD_WEEK)
        assert heating_degree_days(COLD_WEEK) == pytest.approx(expected)

    def test_hot_week_has_none(self) -> None:
        assert heating_degree_days(HOT_WEEK) == 0.0

    def test_at_base_temperature_has_none(self) -> None:
        assert heating_degree_days(MILD_WEEK) == 0.0

    def test_empty_series_is_zero(self) -> None:
        assert heating_degree_days([]) == 0.0

    def test_higher_base_yields_more_degree_days(self) -> None:
        assert heating_degree_days(COLD_WEEK, 25.0) > heating_degree_days(COLD_WEEK, 15.0)

    def test_days_never_go_negative(self) -> None:
        assert heating_degree_days([50.0]) == 0.0


class TestCoolingDegreeDays:
    def test_hot_week_accumulates(self) -> None:
        expected = sum(t - DEFAULT_BASE_TEMPERATURE_C for t in HOT_WEEK)
        assert cooling_degree_days(HOT_WEEK) == pytest.approx(expected)

    def test_cold_week_has_none(self) -> None:
        assert cooling_degree_days(COLD_WEEK) == 0.0

    def test_at_base_temperature_has_none(self) -> None:
        assert cooling_degree_days(MILD_WEEK) == 0.0

    def test_empty_series_is_zero(self) -> None:
        assert cooling_degree_days([]) == 0.0

    def test_lower_base_yields_more_degree_days(self) -> None:
        assert cooling_degree_days(HOT_WEEK, 15.0) > cooling_degree_days(HOT_WEEK, 25.0)

    def test_complements_heating_at_same_base(self) -> None:
        # A day is either a heating day or a cooling day, never both.
        for temp in (0.0, 10.0, 18.0, 25.0, 40.0):
            hdd = heating_degree_days([temp])
            cdd = cooling_degree_days([temp])
            assert hdd == 0.0 or cdd == 0.0


class TestNormalizationFactor:
    def test_identical_periods_give_unity(self) -> None:
        assert normalization_factor(500.0, 500.0) == pytest.approx(1.0)

    def test_milder_current_scales_up(self) -> None:
        assert normalization_factor(600.0, 400.0) > 1.0

    def test_harsher_current_scales_down(self) -> None:
        assert normalization_factor(400.0, 600.0) < 1.0

    def test_zero_current_defaults_to_unity(self) -> None:
        assert normalization_factor(500.0, 0.0) == 1.0

    @pytest.mark.parametrize(("baseline", "current"), [(-1.0, 100.0), (100.0, -1.0)])
    def test_negative_degree_days_rejected(self, baseline: float, current: float) -> None:
        with pytest.raises(ValueError, match="degree days must be non-negative"):
            normalization_factor(baseline, current)


class TestNormalizeConsumption:
    def test_identical_weather_leaves_value_unchanged(self) -> None:
        assert normalize_consumption(1000.0, 500.0, 500.0) == pytest.approx(1000.0)

    def test_mild_current_period_scales_consumption_up(self) -> None:
        assert normalize_consumption(900.0, 600.0, 400.0) > 900.0

    def test_zero_consumption_stays_zero(self) -> None:
        assert normalize_consumption(0.0, 500.0, 300.0) == 0.0

    def test_negative_consumption_rejected(self) -> None:
        with pytest.raises(ValueError, match="consumption_kwh must be non-negative"):
            normalize_consumption(-5.0, 500.0, 500.0)


class TestComparePeriods:
    def test_identical_weather_makes_adjusted_equal_raw(self) -> None:
        result = compare_periods(1000.0, 900.0, 500.0, 500.0)
        assert result.raw_change_pct == pytest.approx(-10.0)
        assert result.normalized_change_pct == pytest.approx(-10.0)
        assert result.weather_effect_pct == pytest.approx(0.0)

    def test_mild_year_masks_efficiency_regression(self) -> None:
        # Consumption fell 10%, but the current period was much milder.
        result = compare_periods(1000.0, 900.0, 500.0, 400.0)
        assert result.raw_change_pct < 0
        assert result.normalized_change_pct > 0
        assert result.weather_effect_pct < 0

    def test_components_sum_to_raw_change(self) -> None:
        result = compare_periods(1200.0, 1000.0, 700.0, 550.0)
        assert result.raw_change_pct == pytest.approx(
            result.normalized_change_pct + result.weather_effect_pct, abs=0.01
        )

    def test_no_change_reports_zero(self) -> None:
        result = compare_periods(800.0, 800.0, 450.0, 450.0)
        assert result.raw_change_pct == pytest.approx(0.0)
        assert result.normalized_change_pct == pytest.approx(0.0)

    def test_zero_current_consumption_allowed(self) -> None:
        result = compare_periods(500.0, 0.0, 300.0, 300.0)
        assert result.normalized_current_kwh == 0.0
        assert result.raw_change_pct == pytest.approx(-100.0)

    def test_non_positive_baseline_rejected(self) -> None:
        with pytest.raises(ValueError, match="baseline_kwh must be positive"):
            compare_periods(0.0, 900.0, 500.0, 500.0)

    def test_negative_current_rejected(self) -> None:
        with pytest.raises(ValueError, match="current_kwh must be non-negative"):
            compare_periods(1000.0, -10.0, 500.0, 500.0)


class TestHeatingDegreeDaysEdgeCases:
    @pytest.mark.parametrize("base", [10.0, 15.0, 18.0, 21.0])
    def test_various_base_temperatures(self, base: float) -> None:
        temps = [base - 5.0] * 7
        result = heating_degree_days(temps, base)
        assert result == pytest.approx(7 * 5.0)

    def test_single_day(self) -> None:
        assert heating_degree_days([10.0]) == pytest.approx(8.0)

    def test_mixed_above_below_base(self) -> None:
        temps = [10.0, 20.0, 10.0]
        result = heating_degree_days(temps, DEFAULT_BASE_TEMPERATURE_C)
        expected = (8.0 + 0.0 + 8.0)
        assert result == pytest.approx(expected)


class TestCoolingDegreeDaysEdgeCases:
    @pytest.mark.parametrize("base", [10.0, 15.0, 18.0, 21.0])
    def test_various_base_temperatures(self, base: float) -> None:
        temps = [base + 5.0] * 7
        result = cooling_degree_days(temps, base)
        assert result == pytest.approx(7 * 5.0)

    def test_single_day(self) -> None:
        assert cooling_degree_days([30.0]) == pytest.approx(12.0)

    def test_exactly_at_base_yields_zero(self) -> None:
        assert cooling_degree_days([DEFAULT_BASE_TEMPERATURE_C]) == 0.0


class TestNormalizationFactorEdgeCases:
    @pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
    def test_expected_scale(self, factor: float) -> None:
        result = normalization_factor(500.0 * factor, 500.0)
        assert result == pytest.approx(factor, rel=1e-3)

    def test_both_zero_returns_unity(self) -> None:
        assert normalization_factor(0.0, 0.0) == 1.0
