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


@pytest.mark.parametrize("annual_kwh,sqm,expected", [
    (17600.0, 100.0, 176.0),
    (5000.0, 50.0, 100.0),
    (25000.0, 125.0, 200.0),
])
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
    from app.benchmarks import benchmark_eui, ASHRAE_EUI_BENCHMARKS
    result = benchmark_eui(200.0, building_type)
    assert result["benchmark_eui"] == ASHRAE_EUI_BENCHMARKS[building_type]
    assert result["rating"] in ("excellent", "good", "average", "poor")
