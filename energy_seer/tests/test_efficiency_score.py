"""Tests for Energy-Seer building efficiency scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.efficiency_score import BUILDING_BENCHMARKS, compute_efficiency_score


def test_compute_returns_dict():
    result = compute_efficiency_score(100.0, "residential")
    assert isinstance(result, dict)


def test_compute_has_required_keys():
    result = compute_efficiency_score(100.0, "residential")
    for key in ("score", "grade", "consumption_per_sqm", "benchmark_per_sqm", "building_type"):
        assert key in result


def test_score_in_range():
    result = compute_efficiency_score(100.0, "residential")
    assert 0.0 <= result["score"] <= 100.0


def test_grade_is_valid_letter():
    result = compute_efficiency_score(100.0, "residential")
    assert result["grade"] in ("A", "B", "C", "D")


def test_low_consumption_gives_high_score():
    result = compute_efficiency_score(1.0, "residential", floor_area_sqm=100.0)
    assert result["score"] > 50


def test_high_consumption_gives_low_score():
    result = compute_efficiency_score(10000.0, "residential", floor_area_sqm=10.0)
    assert result["score"] < 50


def test_negative_consumption_raises():
    with pytest.raises(ValueError):
        compute_efficiency_score(-1.0, "residential")


def test_zero_floor_area_raises():
    with pytest.raises(ValueError):
        compute_efficiency_score(100.0, "residential", floor_area_sqm=0.0)


def test_unknown_building_type_uses_fallback():
    result = compute_efficiency_score(100.0, "unknown_type")
    assert isinstance(result["score"], float)


def test_building_type_normalized():
    result = compute_efficiency_score(100.0, "RESIDENTIAL")
    assert "residential" in result["building_type"].lower()


@pytest.mark.parametrize("building_type", list(BUILDING_BENCHMARKS.keys()))
def test_all_known_building_types(building_type):
    result = compute_efficiency_score(50.0, building_type, floor_area_sqm=100.0)
    assert 0.0 <= result["score"] <= 100.0


def test_grade_a_threshold():
    result = compute_efficiency_score(0.1, "residential", floor_area_sqm=100.0)
    assert result["grade"] == "A"
