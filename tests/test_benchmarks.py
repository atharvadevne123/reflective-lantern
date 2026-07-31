"""Tests for app/benchmarks.py."""

from __future__ import annotations

import pytest


def test_compute_eui_basic():
    from app.benchmarks import compute_eui

    result = compute_eui(10000.0, 100.0)
    assert abs(result - 100.0) < 0.01


def test_compute_eui_zero_area_raises():
    from app.benchmarks import compute_eui

    with pytest.raises(ValueError, match="floor_area_sqm"):
        compute_eui(1000.0, 0.0)


def test_compute_eui_negative_area_raises():
    from app.benchmarks import compute_eui

    with pytest.raises(ValueError, match="floor_area_sqm"):
        compute_eui(1000.0, -50.0)


def test_compute_eui_negative_kwh_raises():
    from app.benchmarks import compute_eui

    with pytest.raises(ValueError, match="annual_kwh"):
        compute_eui(-100.0, 50.0)


def test_compute_eui_zero_kwh():
    from app.benchmarks import compute_eui

    assert compute_eui(0.0, 100.0) == 0.0


@pytest.mark.parametrize(
    "annual_kwh,sqm,expected",
    [
        (17600.0, 100.0, 176.0),
        (5000.0, 50.0, 100.0),
        (25000.0, 125.0, 200.0),
    ],
)
def test_compute_eui_parametrized(annual_kwh, sqm, expected):
    from app.benchmarks import compute_eui

    assert abs(compute_eui(annual_kwh, sqm) - expected) < 0.01


def test_benchmark_eui_excellent():
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(100.0, "office")  # 100 vs 176 benchmark -> ratio ~0.57
    assert result["rating"] == "excellent"


def test_benchmark_eui_poor():
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(300.0, "office")  # 300 vs 176 benchmark -> ratio ~1.7
    assert result["rating"] == "poor"


def test_benchmark_eui_average():
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(176.0, "office")  # ratio = 1.0
    assert result["rating"] == "average"


def test_benchmark_eui_unknown_type_uses_default():
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(200.0, "unknown_type")
    assert result["benchmark_eui"] == 200.0  # default benchmark


def test_benchmark_eui_has_all_keys():
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(150.0, "office")
    assert "building_type" in result
    assert "eui" in result
    assert "benchmark_eui" in result
    assert "ratio" in result
    assert "rating" in result


def test_annual_to_monthly_estimate_sums_to_annual():
    from app.benchmarks import annual_to_monthly_estimate

    result = annual_to_monthly_estimate(12000.0)
    assert len(result) == 12
    assert abs(sum(result) - 12000.0) < 1.0


def test_annual_to_monthly_estimate_custom_profile():
    from app.benchmarks import annual_to_monthly_estimate

    profile = [1.0] * 12
    result = annual_to_monthly_estimate(1200.0, profile=profile)
    assert all(abs(v - 100.0) < 0.1 for v in result)


def test_annual_to_monthly_estimate_invalid_profile():
    from app.benchmarks import annual_to_monthly_estimate

    with pytest.raises(ValueError, match="12 elements"):
        annual_to_monthly_estimate(1000.0, profile=[1.0] * 11)


def test_list_building_types():
    from app.benchmarks import list_building_types

    types = list_building_types()
    assert "office" in types
    assert "hospital" in types
    assert types == sorted(types)


@pytest.mark.parametrize("building_type", ["office", "hotel", "school", "warehouse"])
def test_benchmark_eui_known_types(building_type):
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS, benchmark_eui

    result = benchmark_eui(200.0, building_type)
    assert result["benchmark_eui"] == ASHRAE_EUI_BENCHMARKS[building_type]
    assert result["rating"] in ("excellent", "good", "average", "poor")


def test_site_eui_basic() -> None:
    from app.benchmarks import site_eui

    result = site_eui(10000.0, 1076.39)  # ~100 sqm in sqft
    assert result == pytest.approx(100.0, rel=1e-2)


def test_site_eui_zero_area_raises() -> None:
    from app.benchmarks import site_eui

    with pytest.raises(ValueError, match="floor_area_sqft"):
        site_eui(1000.0, 0.0)


def test_site_eui_negative_kwh_raises() -> None:
    from app.benchmarks import site_eui

    with pytest.raises(ValueError, match="annual_kwh"):
        site_eui(-1.0, 500.0)


def test_compare_buildings_ranks_by_eui() -> None:
    from app.benchmarks import compare_buildings

    buildings = [
        {"name": "A", "annual_kwh": 20000.0, "floor_area_sqm": 100.0},
        {"name": "B", "annual_kwh": 10000.0, "floor_area_sqm": 100.0},
    ]
    results = compare_buildings(buildings)
    assert results[0]["name"] == "B"
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2


def test_compare_buildings_empty() -> None:
    from app.benchmarks import compare_buildings

    assert compare_buildings([]) == []


def test_compare_buildings_single_building() -> None:
    from app.benchmarks import compare_buildings

    buildings = [{"name": "X", "annual_kwh": 5000.0, "floor_area_sqm": 50.0}]
    results = compare_buildings(buildings)
    assert len(results) == 1
    assert results[0]["rank"] == 1
    assert "rating" in results[0]


def test_energy_intensity_ratio_basic() -> None:
    from app.benchmarks import energy_intensity_ratio

    assert energy_intensity_ratio(150.0, 200.0) == pytest.approx(0.75)


def test_energy_intensity_ratio_equal() -> None:
    from app.benchmarks import energy_intensity_ratio

    assert energy_intensity_ratio(200.0, 200.0) == pytest.approx(1.0)


def test_energy_intensity_ratio_zero_reference() -> None:
    from app.benchmarks import energy_intensity_ratio

    assert energy_intensity_ratio(150.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "actual,ref,expected",
    [
        (100.0, 200.0, 0.5),
        (200.0, 100.0, 2.0),
        (0.0, 100.0, 0.0),
    ],
)
def test_energy_intensity_ratio_parametrized(actual: float, ref: float, expected: float) -> None:
    from app.benchmarks import energy_intensity_ratio

    assert energy_intensity_ratio(actual, ref) == pytest.approx(expected, rel=1e-4)


class TestEfficiencyGap:
    def test_over_target(self) -> None:
        # actual 120, target 100 → 20% over
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(120.0, 100.0) == pytest.approx(20.0, rel=1e-4)

    def test_under_target(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(80.0, 100.0) == pytest.approx(-20.0, rel=1e-4)

    def test_equal_values(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(100.0, 100.0) == pytest.approx(0.0, abs=1e-9)

    def test_zero_target_returns_zero(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(50.0, 0.0) == 0.0

    def test_zero_actual(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(0.0, 100.0) == pytest.approx(-100.0, rel=1e-4)

    def test_large_gap(self) -> None:
        from app.benchmarks import efficiency_gap

        result = efficiency_gap(500.0, 200.0)
        assert result == pytest.approx(150.0, rel=1e-4)


def test_list_building_types_not_empty() -> None:
    from app.benchmarks import list_building_types

    types = list_building_types()
    assert isinstance(types, list)
    assert len(types) > 0


def test_list_building_types_are_strings() -> None:
    from app.benchmarks import list_building_types

    for t in list_building_types():
        assert isinstance(t, str)


def test_site_eui_basic_inline() -> None:
    from app.benchmarks import site_eui

    result = site_eui(10000.0, 1000.0)
    assert isinstance(result, float)
    assert result > 0


def test_annual_to_monthly_estimate_length() -> None:
    from app.benchmarks import annual_to_monthly_estimate

    result = annual_to_monthly_estimate(12000.0)
    assert len(result) == 12


def test_annual_to_monthly_estimate_sums_to_annual_approx() -> None:
    from app.benchmarks import annual_to_monthly_estimate

    annual = 12000.0
    result = annual_to_monthly_estimate(annual)
    assert sum(result) == pytest.approx(annual, rel=1e-3)


@pytest.mark.parametrize("building_type", ["hospital", "hotel", "default"])
def test_benchmark_eui_known_type(building_type: str) -> None:
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(100.0, building_type)
    assert isinstance(result, dict)
    assert "benchmark_eui" in result
    assert result["benchmark_eui"] > 0
