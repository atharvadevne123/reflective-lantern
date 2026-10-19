"""Tests for app.thermal_comfort module."""

from __future__ import annotations

import pytest

from app.thermal_comfort import (
    ComfortCategory,
    ThermalComfortResult,
    assess_comfort,
    comfort_category,
    pmv,
    ppd,
    sensation_label,
)


class TestPMV:
    def test_neutral_conditions_near_zero(self) -> None:
        val = pmv(
            air_temp_c=22.0,
            mean_radiant_temp_c=22.0,
            air_velocity_m_s=0.1,
            relative_humidity_pct=50.0,
        )
        assert -1.5 < val < 1.5

    def test_hot_conditions_positive_pmv(self) -> None:
        val = pmv(35.0, 35.0, 0.1, 50.0)
        assert val > 0

    def test_cold_conditions_negative_pmv(self) -> None:
        val = pmv(10.0, 10.0, 0.1, 50.0)
        assert val < 0

    def test_negative_velocity_raises(self) -> None:
        with pytest.raises(ValueError):
            pmv(22.0, 22.0, -0.1, 50.0)

    def test_invalid_humidity_raises(self) -> None:
        with pytest.raises(ValueError):
            pmv(22.0, 22.0, 0.1, 110.0)

    @pytest.mark.parametrize(
        "ta,tr,v,rh",
        [
            (20.0, 20.0, 0.1, 50.0),
            (24.0, 24.0, 0.2, 40.0),
            (18.0, 18.0, 0.15, 60.0),
        ],
    )
    def test_parametrized_returns_float(
        self, ta: float, tr: float, v: float, rh: float
    ) -> None:
        result = pmv(ta, tr, v, rh)
        assert isinstance(result, float)


class TestPPD:
    def test_zero_pmv_gives_minimum_ppd(self) -> None:
        val = ppd(0.0)
        assert val == pytest.approx(5.0, rel=0.01)

    def test_high_pmv_gives_high_ppd(self) -> None:
        assert ppd(3.0) > 90.0

    def test_ppd_always_at_least_five(self) -> None:
        assert ppd(0.0) >= 5.0
        assert ppd(-0.5) >= 5.0

    @pytest.mark.parametrize("pmv_val", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_within_bounds(self, pmv_val: float) -> None:
        val = ppd(pmv_val)
        assert 5.0 <= val <= 100.0


class TestComfortCategory:
    def test_low_ppd_category_i(self) -> None:
        assert comfort_category(5.5) == ComfortCategory.CATEGORY_I

    def test_moderate_ppd_category_ii(self) -> None:
        assert comfort_category(8.0) == ComfortCategory.CATEGORY_II

    def test_higher_ppd_category_iii(self) -> None:
        assert comfort_category(12.0) == ComfortCategory.CATEGORY_III

    def test_very_high_ppd_out_of_range(self) -> None:
        assert comfort_category(50.0) == ComfortCategory.OUT_OF_RANGE

    @pytest.mark.parametrize(
        "ppd_val,expected",
        [
            (5.0, ComfortCategory.CATEGORY_I),
            (9.0, ComfortCategory.CATEGORY_II),
            (14.0, ComfortCategory.CATEGORY_III),
            (30.0, ComfortCategory.OUT_OF_RANGE),
        ],
    )
    def test_parametrized(
        self, ppd_val: float, expected: ComfortCategory
    ) -> None:
        assert comfort_category(ppd_val) == expected


class TestSensationLabel:
    def test_neutral(self) -> None:
        assert sensation_label(0.0) == "Neutral"

    def test_hot(self) -> None:
        assert sensation_label(3.0) == "Hot"

    def test_cold(self) -> None:
        assert sensation_label(-3.0) == "Cold"

    @pytest.mark.parametrize(
        "pmv_val,expected",
        [(-2.0, "Cool"), (-1.0, "Slightly cool"), (1.0, "Slightly warm"), (2.0, "Warm")],
    )
    def test_parametrized(self, pmv_val: float, expected: str) -> None:
        assert sensation_label(pmv_val) == expected


class TestAssessComfort:
    def test_returns_comfort_result(self) -> None:
        result = assess_comfort(22.0, 22.0, 0.1, 50.0)
        assert isinstance(result, ThermalComfortResult)

    def test_ppd_from_pmv_consistent(self) -> None:
        result = assess_comfort(22.0, 22.0, 0.1, 50.0)
        expected_ppd = ppd(result.pmv)
        assert result.ppd == pytest.approx(expected_ppd, rel=1e-4)

    def test_sensation_non_empty(self) -> None:
        result = assess_comfort(22.0, 22.0, 0.1, 50.0)
        assert len(result.sensation) > 0

    @pytest.mark.parametrize(
        "ta,rh",
        [(18.0, 40.0), (22.0, 50.0), (26.0, 60.0)],
    )
    def test_parametrized_conditions(self, ta: float, rh: float) -> None:
        result = assess_comfort(ta, ta, 0.1, rh)
        assert 5.0 <= result.ppd <= 100.0
