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


class TestEfficiencyDelta:
    def test_improvement_positive_delta(self):
        from app.efficiency_score import efficiency_delta
        result = efficiency_delta(60.0, 80.0)
        assert result["delta"] == pytest.approx(20.0)
        assert result["direction"] == "improved"

    def test_degradation_negative_delta(self):
        from app.efficiency_score import efficiency_delta
        result = efficiency_delta(80.0, 60.0)
        assert result["direction"] == "degraded"

    def test_unchanged_when_equal(self):
        from app.efficiency_score import efficiency_delta
        result = efficiency_delta(70.0, 70.0)
        assert result["direction"] == "unchanged"
        assert result["delta"] == 0.0

    def test_grade_keys_present(self):
        from app.efficiency_score import efficiency_delta
        result = efficiency_delta(50.0, 85.0)
        assert "grade_before" in result
        assert "grade_after" in result

    def test_grade_upgrade(self):
        from app.efficiency_score import efficiency_delta
        result = efficiency_delta(55.0, 85.0)
        assert result["grade_before"] == "B"
        assert result["grade_after"] == "A"


class TestComputeEfficiencyScoreValidation:
    def test_negative_consumption_raises(self):
        import pytest
        from app.efficiency_score import compute_efficiency_score
        with pytest.raises(ValueError, match="non-negative"):
            compute_efficiency_score(-1.0, "residential")

    def test_zero_floor_area_raises(self):
        import pytest
        from app.efficiency_score import compute_efficiency_score
        with pytest.raises(ValueError, match="positive"):
            compute_efficiency_score(10.0, "office", floor_area_sqm=0.0)

    @pytest.mark.parametrize("building", ["residential", "commercial", "hospital"])
    def test_known_types_return_grade(self, building):
        from app.efficiency_score import compute_efficiency_score
        result = compute_efficiency_score(5.0, building)
        assert result["grade"] in {"A", "B", "C", "D"}
