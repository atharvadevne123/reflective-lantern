"""Extended tests for energy_seer app/efficiency_score.py."""

from __future__ import annotations

import pytest

from app.efficiency_score import (
    BUILDING_BENCHMARKS,
    compute_efficiency_score,
    efficiency_delta,
    efficiency_percentile,
    efficiency_trend,
    normalised_efficiency_index,
)


class TestComputeEfficiencyScoreExtended:
    def test_negative_consumption_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            compute_efficiency_score(-1.0, "residential")

    def test_zero_floor_area_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_efficiency_score(5.0, "office", floor_area_sqm=0.0)

    def test_negative_floor_area_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_efficiency_score(5.0, "office", floor_area_sqm=-10.0)

    def test_case_insensitive_building_type(self) -> None:
        r1 = compute_efficiency_score(10.0, "RESIDENTIAL")
        r2 = compute_efficiency_score(10.0, "residential")
        assert r1["score"] == r2["score"]

    def test_zero_consumption_max_score(self) -> None:
        result = compute_efficiency_score(0.0, "residential")
        assert result["score"] == 100.0

    def test_all_known_building_types(self) -> None:
        for bt in BUILDING_BENCHMARKS:
            result = compute_efficiency_score(10.0, bt)
            assert 0.0 <= result["score"] <= 100.0
            assert result["grade"] in ("A", "B", "C", "D")

    def test_grade_thresholds_respected(self) -> None:
        result_a = compute_efficiency_score(0.001, "residential")
        assert result_a["grade"] == "A"

    def test_consumption_per_sqm_in_result(self) -> None:
        result = compute_efficiency_score(1000.0, "residential", floor_area_sqm=100.0)
        assert result["consumption_per_sqm"] == pytest.approx(10.0)

    def test_benchmark_per_sqm_in_result(self) -> None:
        result = compute_efficiency_score(5.0, "residential")
        assert result["benchmark_per_sqm"] == pytest.approx(BUILDING_BENCHMARKS["residential"] / 100.0)

    @pytest.mark.parametrize("floor_area", [10.0, 100.0, 1000.0, 10000.0])
    def test_various_floor_areas(self, floor_area: float) -> None:
        result = compute_efficiency_score(500.0, "commercial", floor_area_sqm=floor_area)
        assert 0.0 <= result["score"] <= 100.0


class TestEfficiencyDeltaExtended:
    def test_improved_direction(self) -> None:
        result = efficiency_delta(50.0, 70.0)
        assert result["direction"] == "improved"
        assert result["delta"] == pytest.approx(20.0)

    def test_degraded_direction(self) -> None:
        result = efficiency_delta(70.0, 50.0)
        assert result["direction"] == "degraded"
        assert result["delta"] == pytest.approx(-20.0)

    def test_unchanged_direction(self) -> None:
        result = efficiency_delta(60.0, 60.0)
        assert result["direction"] == "unchanged"
        assert result["delta"] == pytest.approx(0.0)

    def test_grade_before_and_after_in_result(self) -> None:
        result = efficiency_delta(85.0, 65.0)
        assert result["grade_before"] == "A"
        assert result["grade_after"] == "B"

    @pytest.mark.parametrize(
        "before,after,direction",
        [(30.0, 50.0, "improved"), (80.0, 20.0, "degraded"), (60.0, 60.0, "unchanged")],
    )
    def test_direction_parametrized(self, before: float, after: float, direction: str) -> None:
        assert efficiency_delta(before, after)["direction"] == direction


class TestEfficiencyTrendExtended:
    def test_empty_returns_stable(self) -> None:
        result = efficiency_trend([])
        assert result["trend"] == "stable"
        assert result["mean"] == 0.0

    def test_single_value_stable(self) -> None:
        result = efficiency_trend([70.0])
        assert result["trend"] == "stable"
        assert result["mean"] == pytest.approx(70.0)

    def test_improving_trend(self) -> None:
        result = efficiency_trend([50.0, 60.0, 70.0, 80.0])
        assert result["trend"] == "improving"

    def test_declining_trend(self) -> None:
        result = efficiency_trend([80.0, 70.0, 60.0, 50.0])
        assert result["trend"] == "declining"

    def test_stable_trend_small_change(self) -> None:
        result = efficiency_trend([60.0, 60.5, 60.0, 60.5])
        assert result["trend"] == "stable"

    def test_min_max_in_result(self) -> None:
        scores = [40.0, 60.0, 80.0]
        result = efficiency_trend(scores)
        assert result["min"] == pytest.approx(40.0)
        assert result["max"] == pytest.approx(80.0)

    def test_latest_grade_matches_last_score(self) -> None:
        result = efficiency_trend([50.0, 60.0, 85.0])
        assert result["latest_grade"] == "A"


class TestNormalisedEfficiencyIndexExtended:
    def test_zero_floor_area_returns_zero(self) -> None:
        assert normalised_efficiency_index(100.0, "residential", 0.0) == 0.0

    def test_zero_occupancy_hours_returns_zero(self) -> None:
        assert normalised_efficiency_index(100.0, "residential", 100.0, occupancy_hours=0.0) == 0.0

    def test_index_in_unit_interval(self) -> None:
        idx = normalised_efficiency_index(10.0, "residential", 100.0)
        assert 0.0 <= idx <= 1.0

    def test_low_consumption_gives_high_index(self) -> None:
        idx = normalised_efficiency_index(0.001, "residential", 100.0)
        assert idx > 0.5

    @pytest.mark.parametrize("bt", list(BUILDING_BENCHMARKS.keys())[:4])
    def test_various_building_types(self, bt: str) -> None:
        idx = normalised_efficiency_index(10.0, bt, 100.0)
        assert 0.0 <= idx <= 1.0


class TestEfficiencyPercentileExtended:
    def test_empty_population_returns_zero(self) -> None:
        assert efficiency_percentile(70.0, []) == 0.0

    def test_all_below_returns_100(self) -> None:
        assert efficiency_percentile(100.0, [10.0, 20.0, 30.0]) == pytest.approx(100.0)

    def test_all_above_returns_zero(self) -> None:
        assert efficiency_percentile(0.0, [10.0, 20.0, 30.0]) == pytest.approx(0.0)

    def test_median_score_near_50_pct(self) -> None:
        population = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        pct = efficiency_percentile(55.0, population)
        assert 40.0 <= pct <= 60.0

    @pytest.mark.parametrize("score", [0.0, 50.0, 100.0])
    def test_percentile_in_range(self, score: float) -> None:
        population = [float(i * 10) for i in range(11)]
        pct = efficiency_percentile(score, population)
        assert 0.0 <= pct <= 100.0
