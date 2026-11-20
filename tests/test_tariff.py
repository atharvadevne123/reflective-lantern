"""Tests for app/tariff.py."""

from __future__ import annotations

import pytest

from app.tariff import (
    DEFAULT_OFF_PEAK_RATE,
    DEFAULT_PEAK_HOURS,
    DEFAULT_PEAK_RATE,
    TieredBand,
    compare_tariffs,
    flat_rate_cost,
    peak_shift_saving,
    tiered_cost,
    time_of_use_cost,
)

FLAT_DAY = [1.0] * 24


class TestFlatRateCost:
    def test_uniform_day(self) -> None:
        assert flat_rate_cost(FLAT_DAY, rate=0.10) == pytest.approx(2.40)

    def test_empty_series_is_free(self) -> None:
        assert flat_rate_cost([], rate=0.15) == 0.0

    def test_zero_rate_is_free(self) -> None:
        assert flat_rate_cost(FLAT_DAY, rate=0.0) == 0.0

    @pytest.mark.parametrize("rate", [-0.01, -1.0])
    def test_negative_rate_rejected(self, rate: float) -> None:
        with pytest.raises(ValueError, match="rate must be non-negative"):
            flat_rate_cost(FLAT_DAY, rate=rate)

    def test_negative_consumption_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            flat_rate_cost([1.0, -2.0, 3.0])


class TestTimeOfUseCost:
    def test_off_peak_only_uses_off_peak_rate(self) -> None:
        # Hours 0-3 are never in DEFAULT_PEAK_HOURS.
        cost = time_of_use_cost([1.0] * 4, start_hour=0)
        assert cost == pytest.approx(round(4 * DEFAULT_OFF_PEAK_RATE, 2))

    def test_all_peak_uses_peak_rate(self) -> None:
        n = len(DEFAULT_PEAK_HOURS)
        cost = time_of_use_cost([1.0] * n, start_hour=DEFAULT_PEAK_HOURS[0])
        assert cost == pytest.approx(round(n * DEFAULT_PEAK_RATE, 2))

    def test_wraps_past_midnight(self) -> None:
        # Starting at 23 with 3 entries covers hours 23, 0, 1 — all off-peak.
        cost = time_of_use_cost([1.0] * 3, start_hour=23)
        assert cost == pytest.approx(round(3 * DEFAULT_OFF_PEAK_RATE, 2))

    def test_peak_costs_more_than_off_peak(self) -> None:
        peak = time_of_use_cost([1.0] * 5, start_hour=DEFAULT_PEAK_HOURS[0])
        off_peak = time_of_use_cost([1.0] * 5, start_hour=0)
        assert peak > off_peak

    @pytest.mark.parametrize("start_hour", [-1, 24, 99])
    def test_invalid_start_hour_rejected(self, start_hour: int) -> None:
        with pytest.raises(ValueError, match="start_hour must be in 0-23"):
            time_of_use_cost(FLAT_DAY, start_hour=start_hour)

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="rates must be non-negative"):
            time_of_use_cost(FLAT_DAY, peak_rate=-0.5)


class TestTieredCost:
    def test_single_band_is_flat(self) -> None:
        bands = [TieredBand(limit_kwh=None, rate=0.20)]
        assert tiered_cost([10.0] * 10, bands=bands) == pytest.approx(20.0)

    def test_consumption_inside_first_band(self) -> None:
        bands = [TieredBand(limit_kwh=100.0, rate=0.10), TieredBand(limit_kwh=None, rate=0.50)]
        assert tiered_cost([50.0], bands=bands) == pytest.approx(5.0)

    def test_consumption_spans_two_bands(self) -> None:
        bands = [TieredBand(limit_kwh=100.0, rate=0.10), TieredBand(limit_kwh=None, rate=0.50)]
        # 100 kWh at 0.10 + 50 kWh at 0.50 = 10 + 25 = 35
        assert tiered_cost([150.0], bands=bands) == pytest.approx(35.0)

    def test_marginal_rate_increases_with_volume(self) -> None:
        small = tiered_cost([100.0])
        large = tiered_cost([2000.0])
        assert large / 2000.0 > small / 100.0

    def test_empty_series_is_free(self) -> None:
        assert tiered_cost([]) == 0.0

    def test_empty_bands_rejected(self) -> None:
        with pytest.raises(ValueError, match="bands must not be empty"):
            tiered_cost([10.0], bands=[])

    def test_bounded_bands_leave_residual(self, caplog: pytest.LogCaptureFixture) -> None:
        bands = [TieredBand(limit_kwh=10.0, rate=0.10)]
        cost = tiered_cost([100.0], bands=bands)
        assert cost == pytest.approx(1.0)
        assert "did not cover" in caplog.text


class TestCompareTariffs:
    def test_reports_all_three_schemes(self) -> None:
        result = compare_tariffs(FLAT_DAY)
        assert result.flat_cost > 0
        assert result.time_of_use_cost > 0
        assert result.tiered_cost > 0
        assert result.hours_priced == 24

    def test_cheapest_scheme_is_the_minimum(self) -> None:
        result = compare_tariffs(FLAT_DAY)
        costs = {
            "flat": result.flat_cost,
            "time_of_use": result.time_of_use_cost,
            "tiered": result.tiered_cost,
        }
        assert costs[result.cheapest_scheme] == min(costs.values())

    def test_saving_is_flat_minus_cheapest(self) -> None:
        result = compare_tariffs(FLAT_DAY)
        costs = {
            "flat": result.flat_cost,
            "time_of_use": result.time_of_use_cost,
            "tiered": result.tiered_cost,
        }
        assert result.saving_vs_flat == pytest.approx(round(result.flat_cost - costs[result.cheapest_scheme], 2))

    def test_saving_is_never_negative(self) -> None:
        assert compare_tariffs(FLAT_DAY).saving_vs_flat >= 0.0


class TestPeakShiftSaving:
    def test_no_shift_no_saving(self) -> None:
        assert peak_shift_saving(FLAT_DAY, shiftable_fraction=0.0) == 0.0

    def test_full_shift_maximises_saving(self) -> None:
        partial = peak_shift_saving(FLAT_DAY, shiftable_fraction=0.5)
        full = peak_shift_saving(FLAT_DAY, shiftable_fraction=1.0)
        assert full > partial

    def test_no_saving_when_off_peak_not_cheaper(self) -> None:
        saving = peak_shift_saving(FLAT_DAY, shiftable_fraction=1.0, peak_rate=0.10, off_peak_rate=0.10)
        assert saving == 0.0

    def test_off_peak_only_profile_has_no_saving(self) -> None:
        # Hours 0-3 carry no peak load to shift.
        assert peak_shift_saving([1.0] * 4, shiftable_fraction=1.0, start_hour=0) == 0.0

    @pytest.mark.parametrize("fraction", [-0.1, 1.1, 2.0])
    def test_invalid_fraction_rejected(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="shiftable_fraction must be in 0-1"):
            peak_shift_saving(FLAT_DAY, shiftable_fraction=fraction)
