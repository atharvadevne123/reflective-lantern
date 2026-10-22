"""Tests for app.emissions_factor module."""

from __future__ import annotations

import pytest

from app.emissions_factor import (
    EmissionResult,
    Region,
    annual_budget_check,
    calculate_co2,
    compare_regions,
    emissions_factor_kg_per_kwh,
    emissions_saved_by_renewables,
)


class TestEmissionsFactor:
    def test_known_uk_factor(self) -> None:
        assert emissions_factor_kg_per_kwh(Region.UK) == pytest.approx(0.233, rel=1e-3)

    def test_france_low_carbon(self) -> None:
        assert emissions_factor_kg_per_kwh(Region.FR) < 0.1

    @pytest.mark.parametrize(
        "region",
        list(Region),
    )
    def test_all_regions_positive(self, region: Region) -> None:
        assert emissions_factor_kg_per_kwh(region) > 0


class TestCalculateCO2:
    def test_zero_energy_gives_zero_co2(self) -> None:
        result = calculate_co2(0.0, Region.UK)
        assert result.co2_kg == 0.0
        assert result.co2_tonnes == 0.0

    def test_returns_emission_result(self) -> None:
        result = calculate_co2(100.0, Region.UK)
        assert isinstance(result, EmissionResult)
        assert result.region == "UK"

    def test_co2_proportional_to_energy(self) -> None:
        r1 = calculate_co2(100.0, Region.DE)
        r2 = calculate_co2(200.0, Region.DE)
        assert r2.co2_kg == pytest.approx(2 * r1.co2_kg, rel=1e-6)

    def test_negative_energy_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_co2(-10.0, Region.UK)

    def test_tonnes_vs_kg(self) -> None:
        result = calculate_co2(1000.0, Region.UK)
        assert result.co2_tonnes == pytest.approx(result.co2_kg / 1000.0, rel=1e-6)

    @pytest.mark.parametrize(
        "region,energy_kwh",
        [
            (Region.UK, 500.0),
            (Region.FR, 1000.0),
            (Region.AU_NEM, 250.0),
        ],
    )
    def test_parametrized_regions(self, region: Region, energy_kwh: float) -> None:
        result = calculate_co2(energy_kwh, region)
        expected_kg = energy_kwh * emissions_factor_kg_per_kwh(region)
        assert result.co2_kg == pytest.approx(expected_kg, rel=1e-5)


class TestCompareRegions:
    def test_sorted_by_intensity(self) -> None:
        regions = [Region.UK, Region.FR, Region.DE]
        results = compare_regions(1000.0, regions)
        intensities = [r.emissions_factor_kg_per_kwh for r in results]
        assert intensities == sorted(intensities)

    def test_single_region(self) -> None:
        results = compare_regions(100.0, [Region.UK])
        assert len(results) == 1

    def test_empty_regions(self) -> None:
        results = compare_regions(100.0, [])
        assert results == []


class TestEmissionsSavedByRenewables:
    def test_full_displacement(self) -> None:
        dirty = 100.0
        clean = 100.0
        saved = emissions_saved_by_renewables(dirty, clean, Region.UK)
        expected = dirty * emissions_factor_kg_per_kwh(Region.UK)
        assert saved == pytest.approx(expected, rel=1e-4)

    def test_partial_displacement(self) -> None:
        saved = emissions_saved_by_renewables(100.0, 50.0, Region.UK)
        expected = 50.0 * emissions_factor_kg_per_kwh(Region.UK)
        assert saved == pytest.approx(expected, rel=1e-4)

    def test_excess_clean_capped(self) -> None:
        saved_full = emissions_saved_by_renewables(100.0, 100.0, Region.DE)
        saved_excess = emissions_saved_by_renewables(100.0, 200.0, Region.DE)
        assert saved_full == pytest.approx(saved_excess, rel=1e-6)

    def test_negative_dirty_raises(self) -> None:
        with pytest.raises(ValueError):
            emissions_saved_by_renewables(-10.0, 5.0, Region.UK)

    def test_negative_clean_raises(self) -> None:
        with pytest.raises(ValueError):
            emissions_saved_by_renewables(10.0, -5.0, Region.UK)


class TestAnnualBudgetCheck:
    def test_within_large_budget(self) -> None:
        result = annual_budget_check(1000.0, Region.UK, 10000.0)
        assert result["within_budget"] is True

    def test_exceeds_small_budget(self) -> None:
        result = annual_budget_check(10000.0, Region.UK, 1.0)
        assert result["within_budget"] is False

    def test_headroom_calculated(self) -> None:
        budget = 500.0
        result = annual_budget_check(100.0, Region.FR, budget)
        assert result["headroom_kg"] == pytest.approx(budget - result["projected_kg"], rel=1e-5)

    @pytest.mark.parametrize(
        "kwh,region,budget",
        [
            (100.0, Region.UK, 50.0),
            (500.0, Region.FR, 1000.0),
            (1000.0, Region.DE, 300.0),
        ],
    )
    def test_parametrized_budget(self, kwh: float, region: Region, budget: float) -> None:
        result = annual_budget_check(kwh, region, budget)
        projected = calculate_co2(kwh, region).co2_kg
        assert result["within_budget"] is (projected <= budget)
