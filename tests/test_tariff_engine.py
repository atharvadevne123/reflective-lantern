"""Tests for app.tariff_engine module."""

from __future__ import annotations

import pytest

from app.tariff_engine import (
    DEFAULT_TOU_RATES,
    BillEstimate,
    RatePeriod,
    classify_hour,
    peak_shifting_savings,
    rate_for_hour,
    simulate_bill,
)


class TestClassifyHour:
    def test_midnight_off_peak(self) -> None:
        assert classify_hour(0, DEFAULT_TOU_RATES) == RatePeriod.OFF_PEAK

    def test_peak_hour(self) -> None:
        assert classify_hour(17, DEFAULT_TOU_RATES) == RatePeriod.PEAK

    def test_shoulder_morning(self) -> None:
        assert classify_hour(10, DEFAULT_TOU_RATES) == RatePeriod.SHOULDER

    def test_invalid_hour_raises(self) -> None:
        with pytest.raises(ValueError):
            classify_hour(24, DEFAULT_TOU_RATES)

    def test_negative_hour_raises(self) -> None:
        with pytest.raises(ValueError):
            classify_hour(-1, DEFAULT_TOU_RATES)

    @pytest.mark.parametrize(
        "hour,expected",
        [(0, RatePeriod.OFF_PEAK), (7, RatePeriod.SHOULDER), (16, RatePeriod.PEAK)],
    )
    def test_parametrized(self, hour: int, expected: RatePeriod) -> None:
        assert classify_hour(hour, DEFAULT_TOU_RATES) == expected


class TestRateForHour:
    def test_off_peak_rate(self) -> None:
        assert rate_for_hour(3, DEFAULT_TOU_RATES) == pytest.approx(0.08)

    def test_peak_rate(self) -> None:
        assert rate_for_hour(18, DEFAULT_TOU_RATES) == pytest.approx(0.30)

    def test_shoulder_rate(self) -> None:
        assert rate_for_hour(8, DEFAULT_TOU_RATES) == pytest.approx(0.15)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            rate_for_hour(25, DEFAULT_TOU_RATES)

    @pytest.mark.parametrize(
        "hour,expected_rate",
        [(1, 0.08), (12, 0.15), (20, 0.30)],
    )
    def test_parametrized(self, hour: int, expected_rate: float) -> None:
        assert rate_for_hour(hour, DEFAULT_TOU_RATES) == pytest.approx(expected_rate)


class TestSimulateBill:
    def _flat_profile(self, kwh: float = 1.0) -> list[float]:
        return [kwh] * 24

    def test_returns_bill_estimate(self) -> None:
        result = simulate_bill(self._flat_profile())
        assert isinstance(result, BillEstimate)

    def test_total_is_energy_plus_standing(self) -> None:
        result = simulate_bill(self._flat_profile())
        assert result.total_cost == pytest.approx(
            result.energy_cost + result.standing_charge, rel=1e-5
        )

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_bill([1.0] * 12)

    def test_zero_consumption_only_standing(self) -> None:
        result = simulate_bill([0.0] * 24)
        assert result.energy_cost == pytest.approx(0.0)
        assert result.total_cost == pytest.approx(result.standing_charge)

    def test_kwh_by_period_positive(self) -> None:
        result = simulate_bill(self._flat_profile(2.0))
        for v in result.kwh_by_period.values():
            assert v >= 0.0

    def test_cost_by_period_keys(self) -> None:
        result = simulate_bill(self._flat_profile())
        assert set(result.kwh_by_period.keys()) == set(result.cost_by_period.keys())

    @pytest.mark.parametrize(
        "kwh_per_hour",
        [0.5, 1.0, 5.0],
    )
    def test_higher_consumption_higher_cost(self, kwh_per_hour: float) -> None:
        low = simulate_bill([0.5] * 24)
        high = simulate_bill([kwh_per_hour] * 24)
        if kwh_per_hour > 0.5:
            assert high.energy_cost > low.energy_cost


class TestPeakShiftingSavings:
    def test_shift_peak_to_off_peak_positive_saving(self) -> None:
        savings = peak_shifting_savings([1.0] * 24, 1.0, from_hour=18, to_hour=3)
        assert savings > 0

    def test_shift_off_peak_to_peak_negative_saving(self) -> None:
        savings = peak_shifting_savings([1.0] * 24, 1.0, from_hour=3, to_hour=18)
        assert savings < 0

    def test_same_period_zero_saving(self) -> None:
        savings = peak_shifting_savings([1.0] * 24, 1.0, from_hour=18, to_hour=19)
        assert savings == pytest.approx(0.0)

    def test_negative_shift_raises(self) -> None:
        with pytest.raises(ValueError):
            peak_shifting_savings([1.0] * 24, -1.0, 18, 3)

    def test_invalid_from_hour_raises(self) -> None:
        with pytest.raises(ValueError):
            peak_shifting_savings([1.0] * 24, 1.0, 25, 3)

    @pytest.mark.parametrize(
        "shift,from_h,to_h",
        [(0.5, 18, 3), (2.0, 17, 2), (1.0, 20, 5)],
    )
    def test_parametrized_savings_positive(
        self, shift: float, from_h: int, to_h: int
    ) -> None:
        savings = peak_shifting_savings([1.0] * 24, shift, from_h, to_h)
        assert savings >= 0.0
