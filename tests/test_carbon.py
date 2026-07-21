"""Tests for app/carbon.py."""

from __future__ import annotations

import pytest


def test_kwh_to_co2_kg_default_region() -> None:
    from app.carbon import kwh_to_co2_kg
    result = kwh_to_co2_kg(100.0)
    assert result > 0.0
    assert result == pytest.approx(35.0, rel=0.01)


def test_kwh_to_co2_kg_pacific_nw() -> None:
    from app.carbon import kwh_to_co2_kg
    result = kwh_to_co2_kg(100.0, region="pacific_nw")
    assert result == pytest.approx(10.0, rel=0.01)


def test_kwh_to_co2_kg_negative_raises() -> None:
    from app.carbon import kwh_to_co2_kg
    with pytest.raises(ValueError, match="non-negative"):
        kwh_to_co2_kg(-5.0)


def test_kwh_to_co2_kg_zero() -> None:
    from app.carbon import kwh_to_co2_kg
    assert kwh_to_co2_kg(0.0) == 0.0


def test_co2_kg_to_tonnes() -> None:
    from app.carbon import co2_kg_to_tonnes
    assert co2_kg_to_tonnes(1000.0) == pytest.approx(1.0, rel=0.001)


def test_trees_equivalent_basic() -> None:
    from app.carbon import trees_equivalent
    result = trees_equivalent(20.0)
    assert result == pytest.approx(1.0, rel=0.01)


def test_trees_equivalent_zero() -> None:
    from app.carbon import trees_equivalent
    assert trees_equivalent(0.0) == 0.0


def test_annual_carbon_report_valid() -> None:
    from app.carbon import annual_carbon_report
    monthly = [100.0] * 12
    result = annual_carbon_report(monthly)
    assert result["total_kwh"] == pytest.approx(1200.0, rel=0.01)
    assert len(result["monthly_co2_kg"]) == 12
    assert result["total_co2_kg"] > 0


def test_annual_carbon_report_wrong_months() -> None:
    from app.carbon import annual_carbon_report
    with pytest.raises(ValueError, match="12 elements"):
        annual_carbon_report([100.0] * 11)


def test_annual_carbon_report_has_trees() -> None:
    from app.carbon import annual_carbon_report
    result = annual_carbon_report([500.0] * 12)
    assert "trees_equivalent" in result
    assert result["trees_equivalent"] >= 0


def test_carbon_savings_basic() -> None:
    from app.carbon import carbon_savings
    result = carbon_savings(actual_kwh=80.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == pytest.approx(20.0, rel=0.01)
    assert result["saved_co2_kg"] > 0


def test_carbon_savings_no_savings() -> None:
    from app.carbon import carbon_savings
    result = carbon_savings(actual_kwh=100.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == 0.0
    assert result["saved_co2_kg"] == 0.0


def test_carbon_savings_actual_exceeds_baseline() -> None:
    from app.carbon import carbon_savings
    result = carbon_savings(actual_kwh=150.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == 0.0


@pytest.mark.parametrize("region", ["northeast", "midwest", "south", "west", "pacific_nw"])
def test_kwh_to_co2_kg_known_regions(region) -> None:
    from app.carbon import GRID_CARBON_INTENSITY, kwh_to_co2_kg
    result = kwh_to_co2_kg(100.0, region=region)
    expected = GRID_CARBON_INTENSITY[region] * 100.0
    assert abs(result - expected) < 0.01


@pytest.mark.parametrize("region,factor", [
    ("northeast", 0.25),
    ("midwest", 0.45),
    ("pacific_nw", 0.10),
    ("texas", 0.40),
    ("default", 0.35),
])
def test_kwh_to_co2_kg_known_regions_parametrized(region: str, factor: float) -> None:
    from app.carbon import kwh_to_co2_kg
    assert kwh_to_co2_kg(100.0, region=region) == pytest.approx(100.0 * factor, rel=1e-4)


def test_kwh_to_co2_kg_case_insensitive() -> None:
    from app.carbon import kwh_to_co2_kg
    assert kwh_to_co2_kg(100.0, region="PACIFIC_NW") == pytest.approx(kwh_to_co2_kg(100.0, region="pacific_nw"))


def test_kwh_to_co2_kg_unknown_region_uses_default() -> None:
    from app.carbon import GRID_CARBON_INTENSITY, kwh_to_co2_kg
    result = kwh_to_co2_kg(100.0, region="atlantis")
    assert result == pytest.approx(100.0 * GRID_CARBON_INTENSITY["default"])


def test_carbon_savings_no_actual_reduction() -> None:
    from app.carbon import carbon_savings
    result = carbon_savings(actual_kwh=100.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == pytest.approx(0.0)
    assert result["saved_co2_kg"] == pytest.approx(0.0)


@pytest.mark.parametrize("kwh", [0.0, 1.0, 100.0, 10000.0])
def test_kwh_to_co2_kg_non_negative_output(kwh: float) -> None:
    from app.carbon import kwh_to_co2_kg
    assert kwh_to_co2_kg(kwh) >= 0.0


@pytest.mark.parametrize("kg,expected_tonnes", [
    (0.0, 0.0),
    (500.0, 0.5),
    (2000.0, 2.0),
])
def test_co2_kg_to_tonnes_parametrized(kg: float, expected_tonnes: float) -> None:
    from app.carbon import co2_kg_to_tonnes
    assert co2_kg_to_tonnes(kg) == pytest.approx(expected_tonnes, rel=1e-4)


def test_annual_carbon_report_returns_region() -> None:
    from app.carbon import annual_carbon_report
    result = annual_carbon_report([100.0] * 12, region="west")
    assert result["region"] == "west"


def test_annual_carbon_report_empty_list_raises() -> None:
    from app.carbon import annual_carbon_report
    with pytest.raises(ValueError):
        annual_carbon_report([])


@pytest.mark.parametrize("actual,baseline,expected_saved", [
    (50.0, 100.0, 50.0),
    (0.0, 100.0, 100.0),
    (100.0, 50.0, 0.0),
])
def test_carbon_savings_parametrized(actual: float, baseline: float, expected_saved: float) -> None:
    from app.carbon import carbon_savings
    result = carbon_savings(actual_kwh=actual, baseline_kwh=baseline)
    assert result["saved_kwh"] == pytest.approx(expected_saved, rel=1e-4)


def test_trees_equivalent_large_value() -> None:
    from app.carbon import trees_equivalent
    result = trees_equivalent(1_000_000.0)
    assert result > 0
    assert isinstance(result, float)


def test_daily_carbon_estimate_basic() -> None:
    from app.carbon import daily_carbon_estimate
    hourly = [10.0] * 24
    result = daily_carbon_estimate(hourly)
    assert result["total_kwh"] == pytest.approx(240.0)
    assert result["total_co2_kg"] > 0
    assert "trees_equivalent" in result


def test_daily_carbon_estimate_partial_day() -> None:
    from app.carbon import daily_carbon_estimate
    result = daily_carbon_estimate([5.0] * 12)
    assert result["total_kwh"] == pytest.approx(60.0)


def test_daily_carbon_estimate_empty() -> None:
    from app.carbon import daily_carbon_estimate
    result = daily_carbon_estimate([])
    assert result["total_kwh"] == 0.0
    assert result["total_co2_kg"] == 0.0


def test_daily_carbon_estimate_too_many_hours_raises() -> None:
    from app.carbon import daily_carbon_estimate
    with pytest.raises(ValueError, match="24"):
        daily_carbon_estimate([1.0] * 25)


@pytest.mark.parametrize("region,expected_gt", [
    ("pacific_nw", 0.0),
    ("midwest", 0.0),
    ("default", 0.0),
])
def test_daily_carbon_estimate_regions(region: str, expected_gt: float) -> None:
    from app.carbon import daily_carbon_estimate
    result = daily_carbon_estimate([10.0] * 8, region=region)
    assert result["total_co2_kg"] > expected_gt


def test_carbon_intensity_by_hour_basic() -> None:
    from app.carbon import carbon_intensity_by_hour
    result = carbon_intensity_by_hour([10.0, 20.0, 30.0])
    assert len(result) == 3
    assert result[0] < result[1] < result[2]


def test_carbon_intensity_by_hour_empty() -> None:
    from app.carbon import carbon_intensity_by_hour
    assert carbon_intensity_by_hour([]) == []


def test_carbon_intensity_by_hour_sums_to_daily() -> None:
    from app.carbon import carbon_intensity_by_hour, kwh_to_co2_kg
    hourly = [5.0] * 24
    result = carbon_intensity_by_hour(hourly)
    total = sum(result)
    expected = kwh_to_co2_kg(120.0)
    assert abs(total - expected) < 0.01


def test_carbon_intensity_by_hour_region() -> None:
    from app.carbon import carbon_intensity_by_hour
    default_result = carbon_intensity_by_hour([10.0])
    pacific_result = carbon_intensity_by_hour([10.0], region="pacific_nw")
    assert pacific_result[0] < default_result[0]


def test_tree_offset_days_basic() -> None:
    from app.carbon import tree_offset_days
    result = tree_offset_days(20.0, num_trees=1)
    assert result == pytest.approx(365.0, rel=0.01)


def test_tree_offset_days_zero_co2() -> None:
    from app.carbon import tree_offset_days
    assert tree_offset_days(0.0) == 0.0


def test_tree_offset_days_more_trees() -> None:
    from app.carbon import tree_offset_days
    single = tree_offset_days(100.0, num_trees=1)
    ten = tree_offset_days(100.0, num_trees=10)
    assert abs(single - 10 * ten) < 0.1


def test_tree_offset_days_invalid_trees_raises() -> None:
    from app.carbon import tree_offset_days
    with pytest.raises(ValueError, match="num_trees"):
        tree_offset_days(50.0, num_trees=0)


@pytest.mark.parametrize("co2,trees", [(10.0, 1), (10.0, 5), (365.0, 1)])
def test_tree_offset_days_parametrized(co2: float, trees: int) -> None:
    from app.carbon import tree_offset_days
    result = tree_offset_days(co2, num_trees=trees)
    assert result > 0
