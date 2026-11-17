"""Extended tests for Energy-Seer efficiency scoring and grading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.efficiency_score import BUILDING_BENCHMARKS, GRADE_THRESHOLDS, compute_efficiency_score


def test_grade_thresholds_a_above_b():
    assert GRADE_THRESHOLDS["A"] > GRADE_THRESHOLDS["B"]


def test_grade_thresholds_b_above_c():
    assert GRADE_THRESHOLDS["B"] > GRADE_THRESHOLDS["C"]


def test_building_benchmarks_has_residential():
    assert "residential" in BUILDING_BENCHMARKS


def test_building_benchmarks_values_positive():
    for bt, val in BUILDING_BENCHMARKS.items():
        assert val > 0, f"benchmark for {bt} should be positive"


def test_residential_very_efficient_gets_a():
    result = compute_efficiency_score(0.1, "residential", floor_area_sqm=100.0)
    assert result["grade"] == "A"


def test_residential_very_inefficient_gets_d():
    result = compute_efficiency_score(1000.0, "residential", floor_area_sqm=100.0)
    assert result["grade"] == "D"


def test_score_higher_for_lower_consumption():
    r1 = compute_efficiency_score(1.0, "office", floor_area_sqm=100.0)
    r2 = compute_efficiency_score(50.0, "office", floor_area_sqm=100.0)
    assert r1["score"] > r2["score"]


def test_larger_floor_area_improves_score():
    r1 = compute_efficiency_score(100.0, "commercial", floor_area_sqm=100.0)
    r2 = compute_efficiency_score(100.0, "commercial", floor_area_sqm=1000.0)
    assert r2["score"] >= r1["score"]


def test_consumption_per_sqm_field():
    result = compute_efficiency_score(100.0, "office", floor_area_sqm=200.0)
    assert "consumption_per_sqm" in result
    assert result["consumption_per_sqm"] == pytest.approx(0.5)


def test_benchmark_per_sqm_field():
    result = compute_efficiency_score(50.0, "office", floor_area_sqm=100.0)
    assert "benchmark_per_sqm" in result
    assert result["benchmark_per_sqm"] == BUILDING_BENCHMARKS["office"]


def test_building_type_returned_normalized():
    result = compute_efficiency_score(50.0, "OFFICE", floor_area_sqm=100.0)
    assert result["building_type"] == "office"


def test_unknown_building_type_uses_fallback():
    result = compute_efficiency_score(10.0, "spaceship", floor_area_sqm=100.0)
    assert result["score"] is not None


@pytest.mark.parametrize("building_type", ["residential", "commercial", "office", "school"])
def test_known_building_types_all_pass(building_type):
    result = compute_efficiency_score(10.0, building_type, floor_area_sqm=100.0)
    assert 0 <= result["score"] <= 100
    assert result["grade"] in ("A", "B", "C", "D")
