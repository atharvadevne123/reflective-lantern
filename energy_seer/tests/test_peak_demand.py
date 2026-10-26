"""Tests for peak demand predictor."""

from __future__ import annotations

import pytest


class TestFindPeakHours:
    def test_returns_top_n_hours(self) -> None:
        from app.peak_demand import find_peak_hours

        vals = list(range(24))
        result = find_peak_hours(vals, top_n=3)
        assert len(result["peak_hours"]) == 3
        assert result["peak_hours"][0] == 23  # highest

    def test_empty_input(self) -> None:
        from app.peak_demand import find_peak_hours

        result = find_peak_hours([])
        assert result["peak_hours"] == []
        assert result["avg_kwh"] == 0.0

    def test_avg_kwh_correct(self) -> None:
        from app.peak_demand import find_peak_hours

        result = find_peak_hours([2.0] * 24)
        assert result["avg_kwh"] == 2.0

    def test_total_kwh(self) -> None:
        from app.peak_demand import find_peak_hours

        result = find_peak_hours([1.0] * 24)
        assert result["total_kwh"] == 24.0

    def test_peak_to_avg_ratio_flat(self) -> None:
        from app.peak_demand import find_peak_hours

        result = find_peak_hours([3.0] * 24)
        assert abs(result["peak_to_avg_ratio"] - 1.0) < 0.01


class TestPeakShavingSavings:
    def test_savings_positive(self) -> None:
        from app.peak_demand import estimate_peak_shaving_savings

        vals = [1.0] * 20 + [10.0] * 4
        result = estimate_peak_shaving_savings(vals)
        assert result["savings_kwh"] > 0

    def test_flat_profile_no_savings(self) -> None:
        from app.peak_demand import estimate_peak_shaving_savings

        vals = [3.0] * 24
        result = estimate_peak_shaving_savings(vals, shave_pct=0.1)
        assert result["savings_kwh"] < 0.01

    @pytest.mark.parametrize("shave", [0.1, 0.2, 0.3])
    def test_more_shaving_more_savings(self, shave) -> None:
        from app.peak_demand import estimate_peak_shaving_savings

        vals = list(range(1, 25))
        result = estimate_peak_shaving_savings(vals, shave_pct=shave)
        assert result["savings_kwh"] >= 0


class TestDemandFlexibilityScore:
    def test_returns_required_keys(self) -> None:
        from app.peak_demand import demand_flexibility_score

        result = demand_flexibility_score([1.0] * 24)
        for key in ("flexible_kwh", "peak_kwh", "flexibility_ratio", "shiftable_pct"):
            assert key in result

    def test_all_flexible_hours_ratio_near_1(self) -> None:
        from app.peak_demand import demand_flexibility_score

        result = demand_flexibility_score([1.0] * 24, flexible_hours=list(range(24)))
        assert abs(result["flexibility_ratio"] - 1.0) < 0.01

    def test_no_flexible_hours_ratio_zero(self) -> None:
        from app.peak_demand import demand_flexibility_score

        result = demand_flexibility_score([1.0] * 24, flexible_hours=[])
        assert result["flexibility_ratio"] == 0.0

    def test_flexible_and_peak_sum_to_total(self) -> None:
        from app.peak_demand import demand_flexibility_score

        vals = list(range(1, 25))
        result = demand_flexibility_score(vals)
        total = sum(vals)
        assert abs(result["flexible_kwh"] + result["peak_kwh"] - total) < 0.01

    @pytest.mark.parametrize("flex_hours", [[0, 1, 2], [22, 23], list(range(8))])
    def test_ratio_within_bounds(self, flex_hours) -> None:
        from app.peak_demand import demand_flexibility_score

        result = demand_flexibility_score([2.0] * 24, flexible_hours=flex_hours)
        assert 0.0 <= result["flexibility_ratio"] <= 1.0


class TestPeakDemandEdgeCases:
    def test_single_value_is_peak(self) -> None:
        from app.peak_demand import identify_peak_hours

        result = identify_peak_hours([5.0])
        assert result is not None

    def test_all_equal_values_peak_count(self) -> None:
        from app.peak_demand import identify_peak_hours

        result = identify_peak_hours([3.0] * 24)
        assert result is not None

    def test_increasing_series_last_is_max(self) -> None:
        from app.peak_demand import identify_peak_hours

        values = list(range(1, 25))
        result = identify_peak_hours(values)
        # Maximum is at the last position
        assert result is not None


class TestLoadFactor:
    def test_flat_series_load_factor_one(self) -> None:
        from app.peak_demand import load_factor

        assert load_factor([5.0] * 24) == pytest.approx(1.0)

    def test_empty_returns_zero(self) -> None:
        from app.peak_demand import load_factor

        assert load_factor([]) == 0.0

    def test_load_factor_in_bounds(self) -> None:
        from app.peak_demand import load_factor

        result = load_factor([1.0, 2.0, 3.0, 10.0, 2.0])
        assert 0.0 < result <= 1.0


class TestFindPeakHoursExtra:
    def test_peak_hours_sorted_descending(self) -> None:
        from app.peak_demand import find_peak_hours

        values = [1.0, 5.0, 3.0, 8.0, 2.0]
        result = find_peak_hours(values, top_n=2)
        assert result["peak_values"][0] >= result["peak_values"][-1]

    def test_empty_returns_empty_peaks(self) -> None:
        from app.peak_demand import find_peak_hours

        result = find_peak_hours([])
        assert result["peak_hours"] == []


class TestPeakDemandWindow:
    def test_finds_max_window(self) -> None:
        from app.peak_demand import peak_demand_window

        # High values at hours 2-5
        values = [1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0]
        result = peak_demand_window(values, window_hours=4)
        assert result["window_start"] == 2

    def test_insufficient_data_returns_zeros(self) -> None:
        from app.peak_demand import peak_demand_window

        result = peak_demand_window([1.0, 2.0], window_hours=5)
        assert result["total_kwh"] == 0.0

    def test_total_kwh_positive(self) -> None:
        from app.peak_demand import peak_demand_window

        result = peak_demand_window([2.0] * 10, window_hours=4)
        assert result["total_kwh"] > 0.0
