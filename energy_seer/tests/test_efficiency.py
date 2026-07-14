"""Tests for building efficiency scorer and forecaster edge cases."""
from __future__ import annotations

import pytest


class TestEfficiencyScore:
    def test_score_range_0_to_100(self):
        from app.efficiency_score import compute_efficiency_score
        result = compute_efficiency_score(4.5, "residential")
        assert 0 <= result["score"] <= 100

    def test_grade_A_for_low_consumption(self):
        from app.efficiency_score import compute_efficiency_score
        result = compute_efficiency_score(0.1, "residential", floor_area_sqm=100.0)
        assert result["grade"] == "A"

    def test_grade_D_for_very_high_consumption(self):
        from app.efficiency_score import compute_efficiency_score
        result = compute_efficiency_score(9999.0, "residential", floor_area_sqm=10.0)
        assert result["grade"] == "D"

    def test_returns_building_type(self):
        from app.efficiency_score import compute_efficiency_score
        result = compute_efficiency_score(5.0, "office")
        assert result["building_type"] == "office"

    @pytest.mark.parametrize("bt", ["residential", "commercial", "industrial", "office"])
    def test_all_building_types(self, bt):
        from app.efficiency_score import compute_efficiency_score
        result = compute_efficiency_score(10.0, bt)
        assert "score" in result
        assert "grade" in result

    def test_unknown_building_type_uses_default(self):
        from app.efficiency_score import compute_efficiency_score
        result = compute_efficiency_score(10.0, "spaceship")
        assert 0 <= result["score"] <= 100


class TestForecasterEdgeCases:
    def test_linear_trend_two_points(self):
        from app.forecaster import linear_trend
        result = linear_trend([1.0, 3.0])
        assert result["slope"] > 0

    def test_linear_trend_single_point(self):
        from app.forecaster import linear_trend
        result = linear_trend([5.0])
        assert result["next"] == 0.0 or isinstance(result["next"], float)

    def test_seasonal_decompose_too_short(self):
        from app.forecaster import seasonal_decompose_simple
        result = seasonal_decompose_simple([1.0, 2.0], period=24)
        assert result["seasonal"] == []

    def test_moving_average_24_steps(self):
        from app.forecaster import moving_average_forecast
        preds = moving_average_forecast([3.0] * 48, steps=24)
        assert len(preds) == 24
        assert all(abs(p - 3.0) < 0.001 for p in preds)
