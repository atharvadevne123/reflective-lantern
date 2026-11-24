"""Extended efficiency score tests with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.efficiency_score import (
    BUILDING_BENCHMARKS,
    GRADE_THRESHOLDS,
    compare_buildings,
    compute_efficiency_score,
    efficiency_percentile,
    score_to_color,
)


def test_compute_efficiency_score_returns_dict_with_keys() -> None:
    result = compute_efficiency_score(1000.0, "residential", 200.0)
    for key in ("score", "grade", "consumption_per_sqm", "benchmark_per_sqm", "building_type"):
        assert key in result


def test_compute_efficiency_score_negative_consumption_raises() -> None:
    with pytest.raises(ValueError):
        compute_efficiency_score(-1.0, "residential", 100.0)


def test_compute_efficiency_score_zero_area_raises() -> None:
    with pytest.raises(ValueError):
        compute_efficiency_score(100.0, "residential", 0.0)


def test_compute_efficiency_score_negative_area_raises() -> None:
    with pytest.raises(ValueError):
        compute_efficiency_score(100.0, "residential", -10.0)


@pytest.mark.parametrize("building_type", list(BUILDING_BENCHMARKS.keys()))
def test_compute_efficiency_score_known_building_types(building_type: str) -> None:
    result = compute_efficiency_score(100.0, building_type, 100.0)
    assert 0.0 <= result["score"] <= 100.0


def test_compute_efficiency_score_unknown_building_type_uses_default() -> None:
    result = compute_efficiency_score(100.0, "unknown_type", 100.0)
    assert 0.0 <= result["score"] <= 100.0


def test_compute_efficiency_score_case_insensitive() -> None:
    result_lower = compute_efficiency_score(100.0, "residential", 100.0)
    result_upper = compute_efficiency_score(100.0, "RESIDENTIAL", 100.0)
    assert result_lower["score"] == result_upper["score"]


def test_compute_efficiency_score_zero_consumption_high_score() -> None:
    result = compute_efficiency_score(0.0, "residential", 100.0)
    assert result["score"] == 100.0


@pytest.mark.parametrize("grade", list(GRADE_THRESHOLDS.keys()) + ["D"])
def test_grade_is_valid(grade: str) -> None:
    assert grade in ("A", "B", "C", "D")


@pytest.mark.parametrize("score,expected_color", [
    (90.0, "green"),
    (70.0, "green"),
    (50.0, "amber"),
    (20.0, "red"),
])
def test_score_to_color_parametrized(score: float, expected_color: str) -> None:
    assert score_to_color(score) == expected_color


def test_efficiency_percentile_empty_returns_zero() -> None:
    assert efficiency_percentile(50.0, []) == 0.0


def test_efficiency_percentile_all_smaller_returns_100() -> None:
    result = efficiency_percentile(100.0, [10.0, 20.0, 30.0])
    assert result == 100.0


def test_efficiency_percentile_all_larger_returns_zero() -> None:
    result = efficiency_percentile(0.0, [10.0, 20.0, 30.0])
    assert result == 0.0


@pytest.mark.parametrize("score,population,expected_range", [
    (50.0, [0.0, 25.0, 50.0, 75.0, 100.0], (30.0, 60.0)),
    (75.0, [0.0, 25.0, 50.0, 75.0, 100.0], (50.0, 80.0)),
])
def test_efficiency_percentile_in_range(
    score: float, population: list, expected_range: tuple
) -> None:
    result = efficiency_percentile(score, population)
    assert expected_range[0] <= result <= expected_range[1]


def test_compare_buildings_empty_input() -> None:
    result = compare_buildings({})
    assert result["ranked"] == []
    assert result["best"] is None
    assert result["worst"] is None


def test_compare_buildings_single_building() -> None:
    result = compare_buildings({"A": 75.0})
    assert result["best"] == "A"
    assert result["worst"] == "A"
    assert len(result["ranked"]) == 1


def test_compare_buildings_ranked_descending() -> None:
    result = compare_buildings({"A": 50.0, "B": 90.0, "C": 30.0})
    scores = [item["score"] for item in result["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert result["best"] == "B"
    assert result["worst"] == "C"


def test_compare_buildings_all_have_grade() -> None:
    scores = {"A": 85.0, "B": 65.0, "C": 45.0, "D": 20.0}
    result = compare_buildings(scores)
    for item in result["ranked"]:
        assert item["grade"] in ("A", "B", "C", "D")
