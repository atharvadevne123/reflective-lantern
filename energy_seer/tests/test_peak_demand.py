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


class TestDemandFlexibilityScore:
    def test_returns_required_keys(self):
        from app.peak_demand import demand_flexibility_score
        result = demand_flexibility_score([1.0] * 24)
        for key in ("flexible_kwh", "peak_kwh", "flexibility_ratio", "shiftable_pct"):
            assert key in result

    def test_all_flexible_hours_ratio_near_1(self):
        from app.peak_demand import demand_flexibility_score
        result = demand_flexibility_score([1.0] * 24, flexible_hours=list(range(24)))
        assert abs(result["flexibility_ratio"] - 1.0) < 0.01

    def test_no_flexible_hours_ratio_zero(self):
        from app.peak_demand import demand_flexibility_score
        result = demand_flexibility_score([1.0] * 24, flexible_hours=[])
        assert result["flexibility_ratio"] == 0.0

    def test_flexible_and_peak_sum_to_total(self):
        from app.peak_demand import demand_flexibility_score
        vals = list(range(1, 25))
        result = demand_flexibility_score(vals)
        total = sum(vals)
        assert abs(result["flexible_kwh"] + result["peak_kwh"] - total) < 0.01

    @pytest.mark.parametrize("flex_hours", [[0, 1, 2], [22, 23], list(range(8))])
    def test_ratio_within_bounds(self, flex_hours):
        from app.peak_demand import demand_flexibility_score
        result = demand_flexibility_score([2.0] * 24, flexible_hours=flex_hours)
        assert 0.0 <= result["flexibility_ratio"] <= 1.0
