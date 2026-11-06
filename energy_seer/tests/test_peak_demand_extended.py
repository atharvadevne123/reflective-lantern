"""Extended tests for energy_seer app/peak_demand.py."""

from __future__ import annotations

import pytest

from app.peak_demand import (
    demand_flexibility_score,
    estimate_peak_shaving_savings,
    find_peak_hours,
    load_factor,
    peak_demand_window,
)


class TestFindPeakHoursExtended:
    def test_single_value_returns_single_peak(self) -> None:
        result = find_peak_hours([7.5], top_n=1)
        assert result["peak_hours"] == [0]
        assert result["peak_values"] == [7.5]

    def test_top_n_greater_than_length_clamped(self) -> None:
        result = find_peak_hours([1.0, 2.0, 3.0], top_n=10)
        assert len(result["peak_hours"]) == 3

    def test_all_equal_values_returns_top_n(self) -> None:
        result = find_peak_hours([5.0] * 24, top_n=3)
        assert len(result["peak_hours"]) == 3

    def test_peak_to_avg_ratio_spike(self) -> None:
        values = [1.0] * 23 + [100.0]
        result = find_peak_hours(values, top_n=1)
        assert result["peak_to_avg_ratio"] > 1.0

    def test_peak_to_avg_ratio_flat_is_one(self) -> None:
        result = find_peak_hours([4.0] * 24, top_n=3)
        assert abs(result["peak_to_avg_ratio"] - 1.0) < 0.01

    def test_total_kwh_matches_sum(self) -> None:
        values = [float(i) for i in range(1, 25)]
        result = find_peak_hours(values, top_n=3)
        assert abs(result["total_kwh"] - sum(values)) < 0.01

    def test_avg_kwh_matches_mean(self) -> None:
        values = [2.0, 4.0, 6.0, 8.0]
        result = find_peak_hours(values, top_n=2)
        assert abs(result["avg_kwh"] - 5.0) < 0.01

    @pytest.mark.parametrize("top_n", [1, 2, 3, 5, 10])
    def test_various_top_n_values(self, top_n: int) -> None:
        values = list(range(1, 25))
        result = find_peak_hours(values, top_n=top_n)
        assert len(result["peak_hours"]) == min(top_n, 24)


class TestEstimatePeakShavingSavingsExtended:
    def test_returns_required_keys(self) -> None:
        result = estimate_peak_shaving_savings([1.0] * 24)
        for key in ("savings_kwh", "savings_cost", "shave_pct", "peak_threshold_kwh"):
            assert key in result

    def test_zero_cost_per_kwh_zero_savings_cost(self) -> None:
        result = estimate_peak_shaving_savings([1.0] * 10 + [5.0] * 2, cost_per_kwh=0.0)
        assert result["savings_cost"] == pytest.approx(0.0)

    def test_savings_kwh_non_negative(self) -> None:
        values = list(range(1, 25))
        result = estimate_peak_shaving_savings(values, shave_pct=0.1)
        assert result["savings_kwh"] >= 0.0

    def test_shave_pct_stored(self) -> None:
        result = estimate_peak_shaving_savings([1.0] * 24, shave_pct=0.2)
        assert result["shave_pct"] == pytest.approx(0.2)

    def test_savings_cost_proportional_to_cost_per_kwh(self) -> None:
        vals = list(range(1, 25))
        r1 = estimate_peak_shaving_savings(vals, cost_per_kwh=0.10)
        r2 = estimate_peak_shaving_savings(vals, cost_per_kwh=0.20)
        assert r2["savings_cost"] == pytest.approx(r1["savings_cost"] * 2, rel=1e-3)

    def test_full_shave_reduces_all_to_threshold(self) -> None:
        values = [1.0] * 20 + [10.0] * 4
        result = estimate_peak_shaving_savings(values, shave_pct=0.0)
        assert result["savings_kwh"] == pytest.approx(0.0, abs=0.001)

    @pytest.mark.parametrize("shave", [0.05, 0.1, 0.15, 0.2, 0.3])
    def test_shave_pct_range(self, shave: float) -> None:
        vals = list(range(1, 25))
        result = estimate_peak_shaving_savings(vals, shave_pct=shave)
        assert result["savings_kwh"] >= 0.0


class TestDemandFlexibilityScoreExtended:
    def test_empty_flexible_hours_zero_ratio(self) -> None:
        result = demand_flexibility_score([2.0] * 24, flexible_hours=[])
        assert result["flexibility_ratio"] == 0.0
        assert result["shiftable_pct"] == 0.0

    def test_all_flexible_ratio_is_one(self) -> None:
        result = demand_flexibility_score([1.0] * 24, flexible_hours=list(range(24)))
        assert abs(result["flexibility_ratio"] - 1.0) < 0.001

    def test_default_flexible_hours_set(self) -> None:
        result = demand_flexibility_score([1.0] * 24)
        assert 0.0 < result["flexibility_ratio"] < 1.0

    def test_flexible_and_peak_sum_to_total_kwh(self) -> None:
        values = [float(i) for i in range(24)]
        result = demand_flexibility_score(values, flexible_hours=[0, 1, 2])
        total = sum(values)
        assert abs(result["flexible_kwh"] + result["peak_kwh"] - total) < 0.001

    def test_shiftable_pct_is_ratio_times_100(self) -> None:
        result = demand_flexibility_score([2.0] * 24, flexible_hours=[0, 1, 2])
        assert abs(result["shiftable_pct"] - result["flexibility_ratio"] * 100) < 0.001

    def test_out_of_bounds_flexible_hours_ignored(self) -> None:
        result = demand_flexibility_score([1.0] * 24, flexible_hours=[0, 25, 100])
        assert result["flexible_kwh"] == pytest.approx(1.0)


class TestPeakDemandWindowExtended:
    def test_window_covers_highest_block(self) -> None:
        values = [1.0] * 4 + [10.0] * 4 + [1.0] * 4
        result = peak_demand_window(values, window_hours=4)
        assert result["window_start"] == 4

    def test_window_end_is_start_plus_window_minus_one(self) -> None:
        values = list(range(10))
        result = peak_demand_window(values, window_hours=3)
        assert result["window_end"] == result["window_start"] + 2

    def test_avg_kwh_equals_total_over_window(self) -> None:
        values = [3.0] * 10
        result = peak_demand_window(values, window_hours=4)
        assert abs(result["avg_kwh"] - result["total_kwh"] / 4) < 0.001

    def test_window_one_hour(self) -> None:
        values = [5.0, 1.0, 1.0, 10.0, 2.0]
        result = peak_demand_window(values, window_hours=1)
        assert result["window_start"] == 3

    @pytest.mark.parametrize("w", [1, 2, 3, 6])
    def test_window_size_varied(self, w: int) -> None:
        values = [float(i) for i in range(24)]
        result = peak_demand_window(values, window_hours=w)
        assert result["total_kwh"] > 0.0


class TestLoadFactorExtended:
    def test_all_zeros_returns_zero(self) -> None:
        assert load_factor([0.0] * 10) == 0.0

    def test_flat_nonzero_returns_one(self) -> None:
        assert load_factor([7.5] * 24) == pytest.approx(1.0)

    def test_highly_variable_less_than_one(self) -> None:
        values = [0.1] * 23 + [100.0]
        lf = load_factor(values)
        assert 0.0 < lf < 1.0

    def test_single_value_returns_one(self) -> None:
        assert load_factor([5.0]) == pytest.approx(1.0)

    @pytest.mark.parametrize("peak_val", [1.0, 10.0, 100.0])
    def test_load_factor_in_unit_interval(self, peak_val: float) -> None:
        values = [1.0] * 10 + [peak_val]
        lf = load_factor(values)
        assert 0.0 <= lf <= 1.0


class TestOffPeakAverageAndRollingPeakSums:
    def test_off_peak_average_empty_returns_zero(self) -> None:
        from app.peak_demand import off_peak_average
        assert off_peak_average([]) == 0.0

    def test_off_peak_average_all_hours_flexible(self) -> None:
        from app.peak_demand import off_peak_average
        values = [3.0] * 24
        result = off_peak_average(values, off_peak_hours=list(range(24)))
        assert result == pytest.approx(3.0)

    def test_off_peak_average_specific_hours(self) -> None:
        from app.peak_demand import off_peak_average
        values = [1.0] * 20 + [5.0] * 4
        result = off_peak_average(values, off_peak_hours=list(range(20)))
        assert result == pytest.approx(1.0)

    def test_off_peak_average_out_of_bounds_hours(self) -> None:
        from app.peak_demand import off_peak_average
        values = [2.0] * 10
        result = off_peak_average(values, off_peak_hours=[0, 100])
        assert result == pytest.approx(2.0)

    def test_rolling_peak_sums_empty_when_too_short(self) -> None:
        from app.peak_demand import rolling_peak_sums
        assert rolling_peak_sums([1.0, 2.0], window_hours=5) == []

    def test_rolling_peak_sums_length(self) -> None:
        from app.peak_demand import rolling_peak_sums
        values = list(range(10))
        result = rolling_peak_sums(values, window_hours=3)
        assert len(result) == 8

    def test_rolling_peak_sums_values(self) -> None:
        from app.peak_demand import rolling_peak_sums
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = rolling_peak_sums(values, window_hours=2)
        assert result[0] == pytest.approx(3.0)
        assert result[-1] == pytest.approx(9.0)

    def test_rolling_peak_sums_single_window(self) -> None:
        from app.peak_demand import rolling_peak_sums
        values = [2.0, 3.0, 4.0]
        result = rolling_peak_sums(values, window_hours=3)
        assert result == [pytest.approx(9.0)]


class TestPeakDemandWindowEdgeCases:
    def test_too_short_returns_zeros(self) -> None:
        result = peak_demand_window([1.0, 2.0], window_hours=4)
        assert result["total_kwh"] == pytest.approx(0.0)
        assert result["window_start"] == 0

    def test_window_start_and_end_correct(self) -> None:
        values = [0.0] * 5 + [10.0, 10.0, 10.0, 10.0] + [0.0] * 5
        result = peak_demand_window(values, window_hours=4)
        assert result["window_start"] == 5
        assert result["window_end"] == 8

    def test_avg_kwh_is_total_divided_by_window(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = peak_demand_window(values, window_hours=3)
        expected_avg = result["total_kwh"] / 3
        assert result["avg_kwh"] == pytest.approx(expected_avg, abs=0.01)

    @pytest.mark.parametrize("window", [2, 3, 4, 6])
    def test_various_window_sizes(self, window: int) -> None:
        values = [float(i) for i in range(24)]
        result = peak_demand_window(values, window_hours=window)
        assert result["total_kwh"] > 0.0

    def test_flat_series_start_at_zero(self) -> None:
        values = [5.0] * 10
        result = peak_demand_window(values, window_hours=4)
        assert result["window_start"] == 0

    def test_window_end_equals_start_plus_window_minus_one(self) -> None:
        values = list(range(1, 13))
        result = peak_demand_window(values, window_hours=4)
        assert result["window_end"] == result["window_start"] + 3
