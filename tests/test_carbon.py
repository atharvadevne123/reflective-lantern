"""Tests for app/carbon.py."""

from __future__ import annotations

import pytest


def test_kwh_to_co2_kg_default_region():
    from app.carbon import kwh_to_co2_kg
    result = kwh_to_co2_kg(100.0)
    assert result > 0.0
    assert result == pytest.approx(35.0, rel=0.01)


def test_kwh_to_co2_kg_pacific_nw():
    from app.carbon import kwh_to_co2_kg
    result = kwh_to_co2_kg(100.0, region="pacific_nw")
    assert result == pytest.approx(10.0, rel=0.01)


def test_kwh_to_co2_kg_negative_raises():
    from app.carbon import kwh_to_co2_kg
    with pytest.raises(ValueError, match="non-negative"):
        kwh_to_co2_kg(-5.0)


def test_kwh_to_co2_kg_zero():
    from app.carbon import kwh_to_co2_kg
    assert kwh_to_co2_kg(0.0) == 0.0


def test_co2_kg_to_tonnes():
    from app.carbon import co2_kg_to_tonnes
    assert co2_kg_to_tonnes(1000.0) == pytest.approx(1.0, rel=0.001)


def test_trees_equivalent_basic():
    from app.carbon import trees_equivalent
    result = trees_equivalent(20.0)
    assert result == pytest.approx(1.0, rel=0.01)


def test_trees_equivalent_zero():
    from app.carbon import trees_equivalent
    assert trees_equivalent(0.0) == 0.0


def test_annual_carbon_report_valid():
    from app.carbon import annual_carbon_report
    monthly = [100.0] * 12
    result = annual_carbon_report(monthly)
    assert result["total_kwh"] == pytest.approx(1200.0, rel=0.01)
    assert len(result["monthly_co2_kg"]) == 12
    assert result["total_co2_kg"] > 0


def test_annual_carbon_report_wrong_months():
    from app.carbon import annual_carbon_report
    with pytest.raises(ValueError, match="12 elements"):
        annual_carbon_report([100.0] * 11)


def test_annual_carbon_report_has_trees():
    from app.carbon import annual_carbon_report
    result = annual_carbon_report([500.0] * 12)
    assert "trees_equivalent" in result
    assert result["trees_equivalent"] >= 0


def test_carbon_savings_basic():
    from app.carbon import carbon_savings
    result = carbon_savings(actual_kwh=80.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == pytest.approx(20.0, rel=0.01)
    assert result["saved_co2_kg"] > 0


def test_carbon_savings_no_savings():
    from app.carbon import carbon_savings
    result = carbon_savings(actual_kwh=100.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == 0.0
    assert result["saved_co2_kg"] == 0.0


def test_carbon_savings_actual_exceeds_baseline():
    from app.carbon import carbon_savings
    result = carbon_savings(actual_kwh=150.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == 0.0


@pytest.mark.parametrize("region", ["northeast", "midwest", "south", "west", "pacific_nw"])
def test_kwh_to_co2_kg_known_regions(region):
    from app.carbon import GRID_CARBON_INTENSITY, kwh_to_co2_kg
    result = kwh_to_co2_kg(100.0, region=region)
    expected = GRID_CARBON_INTENSITY[region] * 100.0
    assert abs(result - expected) < 0.01
