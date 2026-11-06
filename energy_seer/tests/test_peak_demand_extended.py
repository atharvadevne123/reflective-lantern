"""Extended tests for energy_seer app/peak_demand.py."""

from __future__ import annotations

import pytest

from app.peak_demand import find_peak_hours


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
