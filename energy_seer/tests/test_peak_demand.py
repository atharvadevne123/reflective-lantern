"""Tests for peak demand predictor."""
from __future__ import annotations

import pytest


class TestFindPeakHours:
    def test_returns_top_n_hours(self):
        from app.peak_demand import find_peak_hours
        vals = list(range(24))
        result = find_peak_hours(vals, top_n=3)
        assert len(result["peak_hours"]) == 3
        assert result["peak_hours"][0] == 23  # highest

    def test_empty_input(self):
        from app.peak_demand import find_peak_hours
        result = find_peak_hours([])
        assert result["peak_hours"] == []
        assert result["avg_kwh"] == 0.0

    def test_avg_kwh_correct(self):
        from app.peak_demand import find_peak_hours
        result = find_peak_hours([2.0] * 24)
        assert result["avg_kwh"] == 2.0

    def test_total_kwh(self):
        from app.peak_demand import find_peak_hours
        result = find_peak_hours([1.0] * 24)
        assert result["total_kwh"] == 24.0

    def test_peak_to_avg_ratio_flat(self):
        from app.peak_demand import find_peak_hours
        result = find_peak_hours([3.0] * 24)
        assert abs(result["peak_to_avg_ratio"] - 1.0) < 0.01


class TestPeakShavingSavings:
    def test_savings_positive(self):
        from app.peak_demand import estimate_peak_shaving_savings
        vals = [1.0] * 20 + [10.0] * 4
        result = estimate_peak_shaving_savings(vals)
        assert result["savings_kwh"] > 0

    def test_flat_profile_no_savings(self):
        from app.peak_demand import estimate_peak_shaving_savings
        vals = [3.0] * 24
        result = estimate_peak_shaving_savings(vals, shave_pct=0.1)
        assert result["savings_kwh"] < 0.01

    @pytest.mark.parametrize("shave", [0.1, 0.2, 0.3])
    def test_more_shaving_more_savings(self, shave):
        from app.peak_demand import estimate_peak_shaving_savings
        vals = list(range(1, 25))
        result = estimate_peak_shaving_savings(vals, shave_pct=shave)
        assert result["savings_kwh"] >= 0
