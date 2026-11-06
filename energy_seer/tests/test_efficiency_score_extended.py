"""Extended tests for energy_seer app/efficiency_score.py."""

from __future__ import annotations

import pytest

from app.efficiency_score import BUILDING_BENCHMARKS, compute_efficiency_score


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
