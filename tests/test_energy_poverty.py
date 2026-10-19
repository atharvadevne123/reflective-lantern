"""Tests for app.energy_poverty module."""

from __future__ import annotations

import pytest

from app.energy_poverty import (
    FuelPovertyAssessment,
    after_fuel_costs_income,
    assess_household,
    assess_ten_percent,
    energy_cost_ratio,
    recommended_interventions,
    vulnerability_score,
)


class TestEnergyCostRatio:
    def test_ten_percent(self) -> None:
        assert energy_cost_ratio(1000.0, 10000.0) == pytest.approx(0.1)

    def test_zero_cost(self) -> None:
        assert energy_cost_ratio(0.0, 10000.0) == pytest.approx(0.0)

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_cost_ratio(-100.0, 10000.0)

    def test_zero_income_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_cost_ratio(500.0, 0.0)

    @pytest.mark.parametrize(
        "cost,income,expected",
        [(2000.0, 20000.0, 0.1), (500.0, 10000.0, 0.05), (3000.0, 15000.0, 0.2)],
    )
    def test_parametrized(self, cost: float, income: float, expected: float) -> None:
        assert energy_cost_ratio(cost, income) == pytest.approx(expected)


class TestAssessTenPercent:
    def test_above_threshold_is_fuel_poor(self) -> None:
        assert assess_ten_percent(1500.0, 10000.0) is True

    def test_below_threshold_not_fuel_poor(self) -> None:
        assert assess_ten_percent(500.0, 10000.0) is False

    def test_exactly_at_threshold(self) -> None:
        assert assess_ten_percent(1000.0, 10000.0) is True

    @pytest.mark.parametrize(
        "cost,income,expected",
        [(2000.0, 10000.0, True), (100.0, 10000.0, False), (999.0, 10000.0, False)],
    )
    def test_parametrized(self, cost: float, income: float, expected: bool) -> None:
        assert assess_ten_percent(cost, income) is expected


class TestAfterFuelCostsIncome:
    def test_basic(self) -> None:
        assert after_fuel_costs_income(20000.0, 1500.0) == pytest.approx(18500.0)

    def test_costs_exceed_income(self) -> None:
        result = after_fuel_costs_income(1000.0, 2000.0)
        assert result == pytest.approx(-1000.0)

    def test_negative_income_raises(self) -> None:
        with pytest.raises(ValueError):
            after_fuel_costs_income(-100.0, 500.0)

    def test_zero_cost(self) -> None:
        assert after_fuel_costs_income(15000.0, 0.0) == pytest.approx(15000.0)


class TestVulnerabilityScore:
    def test_low_vulnerability(self) -> None:
        score = vulnerability_score(40, False, "B")
        assert score < 50.0

    def test_high_vulnerability(self) -> None:
        score = vulnerability_score(80, True, "G", children_under_5=2)
        assert score > 50.0

    def test_invalid_epc_raises(self) -> None:
        with pytest.raises(ValueError):
            vulnerability_score(50, False, "Z")

    def test_within_bounds(self) -> None:
        score = vulnerability_score(70, True, "F", 1)
        assert 0.0 <= score <= 100.0

    @pytest.mark.parametrize(
        "age,disability,epc",
        [(30, False, "A"), (65, False, "D"), (75, True, "G")],
    )
    def test_parametrized_non_negative(
        self, age: int, disability: bool, epc: str
    ) -> None:
        assert vulnerability_score(age, disability, epc) >= 0.0


class TestRecommendedInterventions:
    def test_returns_list(self) -> None:
        recs = recommended_interventions("D", True, False)
        assert isinstance(recs, list) and len(recs) > 0

    def test_fuel_poor_has_scheme_recommendation(self) -> None:
        recs = recommended_interventions("E", True, False)
        assert any("scheme" in r.lower() or "discount" in r.lower() for r in recs)

    def test_disability_has_priority_services(self) -> None:
        recs = recommended_interventions("C", False, True)
        assert any("priority" in r.lower() for r in recs)

    @pytest.mark.parametrize("epc", ["A", "C", "F", "G"])
    def test_all_epcs_return_non_empty(self, epc: str) -> None:
        recs = recommended_interventions(epc, False, False)
        assert len(recs) > 0


class TestAssessHousehold:
    def test_returns_assessment(self) -> None:
        result = assess_household("H001", 1500.0, 12000.0, 45, False, "D")
        assert isinstance(result, FuelPovertyAssessment)

    def test_household_id_preserved(self) -> None:
        result = assess_household("H99", 2000.0, 15000.0, 60, True, "E")
        assert result.household_id == "H99"

    def test_fuel_poor_flag(self) -> None:
        result = assess_household("X", 2000.0, 10000.0, 40, False, "C")
        assert result.is_fuel_poor is True

    def test_not_fuel_poor(self) -> None:
        result = assess_household("Y", 500.0, 50000.0, 40, False, "B")
        assert result.is_fuel_poor is False

    @pytest.mark.parametrize(
        "cost,income,expected_poor",
        [(3000.0, 15000.0, True), (200.0, 30000.0, False)],
    )
    def test_parametrized(
        self, cost: float, income: float, expected_poor: bool
    ) -> None:
        result = assess_household("Z", cost, income, 35, False, "C")
        assert result.is_fuel_poor is expected_poor
