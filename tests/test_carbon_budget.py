"""Tests for app.carbon_budget module."""

from __future__ import annotations

import pytest

from app.carbon_budget import (
    CarbonBudgetResult,
    EmissionEntry,
    ScopeCategory,
    analyse_budget,
    carbon_intensity,
    emissions_by_scope,
    linear_reduction_pathway,
    reduction_target_kg,
    total_emissions,
)


def _entry(scope: ScopeCategory, kg: float) -> EmissionEntry:
    return EmissionEntry(category="test", scope=scope, kg_co2e=kg)


class TestTotalEmissions:
    def test_sums_correctly(self) -> None:
        entries = [_entry(ScopeCategory.SCOPE_1, 100.0), _entry(ScopeCategory.SCOPE_2, 50.0)]
        assert total_emissions(entries) == pytest.approx(150.0)

    def test_empty_list_zero(self) -> None:
        assert total_emissions([]) == pytest.approx(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            total_emissions([_entry(ScopeCategory.SCOPE_1, -10.0)])

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_parametrized_count(self, n: int) -> None:
        entries = [_entry(ScopeCategory.SCOPE_1, 100.0)] * n
        assert total_emissions(entries) == pytest.approx(100.0 * n)


class TestEmissionsByScope:
    def test_filters_scope(self) -> None:
        entries = [
            _entry(ScopeCategory.SCOPE_1, 200.0),
            _entry(ScopeCategory.SCOPE_2, 100.0),
            _entry(ScopeCategory.SCOPE_3, 50.0),
        ]
        assert emissions_by_scope(entries, ScopeCategory.SCOPE_1) == pytest.approx(200.0)

    def test_empty_scope_zero(self) -> None:
        entries = [_entry(ScopeCategory.SCOPE_1, 100.0)]
        assert emissions_by_scope(entries, ScopeCategory.SCOPE_2) == pytest.approx(0.0)

    @pytest.mark.parametrize("scope", list(ScopeCategory))
    def test_all_scopes(self, scope: ScopeCategory) -> None:
        entries = [_entry(scope, 300.0)]
        assert emissions_by_scope(entries, scope) == pytest.approx(300.0)


class TestReductionTargetKg:
    def test_50_percent(self) -> None:
        assert reduction_target_kg(1000.0, 50.0) == pytest.approx(500.0)

    def test_zero_reduction(self) -> None:
        assert reduction_target_kg(1000.0, 0.0) == pytest.approx(1000.0)

    def test_100_reduction_zero(self) -> None:
        assert reduction_target_kg(1000.0, 100.0) == pytest.approx(0.0)

    def test_negative_baseline_raises(self) -> None:
        with pytest.raises(ValueError):
            reduction_target_kg(-100.0, 50.0)

    def test_invalid_pct_raises(self) -> None:
        with pytest.raises(ValueError):
            reduction_target_kg(1000.0, 110.0)


class TestLinearReductionPathway:
    def test_length(self) -> None:
        pathway = linear_reduction_pathway(1000.0, 0.0, 10)
        assert len(pathway) == 11

    def test_starts_at_baseline(self) -> None:
        pathway = linear_reduction_pathway(1000.0, 0.0, 10)
        assert pathway[0] == pytest.approx(1000.0)

    def test_ends_at_target(self) -> None:
        pathway = linear_reduction_pathway(1000.0, 0.0, 10)
        assert pathway[-1] == pytest.approx(0.0)

    def test_zero_years_raises(self) -> None:
        with pytest.raises(ValueError):
            linear_reduction_pathway(1000.0, 0.0, 0)

    @pytest.mark.parametrize("years", [5, 10, 30])
    def test_monotone_decreasing(self, years: int) -> None:
        pathway = linear_reduction_pathway(1000.0, 0.0, years)
        for i in range(len(pathway) - 1):
            assert pathway[i] >= pathway[i + 1]


class TestCarbonIntensity:
    def test_basic(self) -> None:
        assert carbon_intensity(1000.0, 100.0) == pytest.approx(10.0)

    def test_negative_emissions_raises(self) -> None:
        with pytest.raises(ValueError):
            carbon_intensity(-100.0, 100.0)

    def test_zero_activity_raises(self) -> None:
        with pytest.raises(ValueError):
            carbon_intensity(100.0, 0.0)

    @pytest.mark.parametrize("activity", [1.0, 100.0, 1000.0])
    def test_parametrized(self, activity: float) -> None:
        assert carbon_intensity(1000.0, activity) == pytest.approx(1000.0 / activity)


class TestAnalyseBudget:
    def test_returns_result(self) -> None:
        entries = [_entry(ScopeCategory.SCOPE_1, 500.0)]
        result = analyse_budget(entries, 1000.0)
        assert isinstance(result, CarbonBudgetResult)

    def test_on_track_when_under_budget(self) -> None:
        entries = [_entry(ScopeCategory.SCOPE_1, 500.0)]
        result = analyse_budget(entries, 1000.0)
        assert result.on_track is True

    def test_not_on_track_when_over_budget(self) -> None:
        entries = [_entry(ScopeCategory.SCOPE_1, 1500.0)]
        result = analyse_budget(entries, 1000.0)
        assert result.on_track is False

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_budget([], -100.0)

    @pytest.mark.parametrize("budget", [500.0, 1000.0, 5000.0])
    def test_budget_remaining(self, budget: float) -> None:
        entries = [_entry(ScopeCategory.SCOPE_2, 300.0)]
        result = analyse_budget(entries, budget)
        assert result.budget_remaining_kg == pytest.approx(budget - 300.0)
