"""Tests for fleet-mind app/cost_optimizer.py."""

from __future__ import annotations

import pytest

from app.cost_optimizer import (
    CostModel,
    RepairOption,
    annual_maintenance_budget,
    break_even_km,
    cost_per_reliability_unit,
    preventive_vs_reactive_saving,
    rank_options,
    total_repair_cost,
)


def _option(cost: float, downtime: float = 1.0, gain: float = 0.5) -> RepairOption:
    return RepairOption("desc", cost=cost, downtime_hours=downtime, reliability_gain=gain)


class TestTotalRepairCost:
    def test_includes_downtime(self) -> None:
        model = CostModel(downtime_cost_per_hour=100.0)
        option = _option(cost=500.0, downtime=2.0)
        assert total_repair_cost(option, model) == pytest.approx(700.0)

    def test_zero_downtime(self) -> None:
        model = CostModel()
        option = _option(cost=200.0, downtime=0.0)
        assert total_repair_cost(option, model) == pytest.approx(200.0)


class TestPreventiveVsReactiveSaving:
    def test_saving_when_preventive_cheaper(self) -> None:
        assert preventive_vs_reactive_saving(1000.0, 400.0) == pytest.approx(600.0)

    def test_no_saving_when_reactive_cheaper(self) -> None:
        assert preventive_vs_reactive_saving(300.0, 500.0) == pytest.approx(0.0)


class TestCostPerReliabilityUnit:
    def test_zero_gain_returns_inf(self) -> None:
        option = _option(cost=200.0, gain=0.0)
        assert cost_per_reliability_unit(option, CostModel()) == float("inf")

    def test_positive_gain(self) -> None:
        option = _option(cost=200.0, downtime=0.0, gain=1.0)
        result = cost_per_reliability_unit(option, CostModel())
        assert result == pytest.approx(200.0)


class TestRankOptions:
    def test_cheapest_per_unit_first(self) -> None:
        options = [
            _option(cost=1000.0, downtime=0.0, gain=1.0),
            _option(cost=100.0, downtime=0.0, gain=1.0),
        ]
        ranked = rank_options(options, CostModel())
        assert ranked[0].cost == pytest.approx(100.0)


class TestAnnualMaintenanceBudget:
    def test_includes_contingency(self) -> None:
        budget = annual_maintenance_budget(10, 1000.0, contingency_pct=0.10)
        assert budget == pytest.approx(11_000.0)


class TestBreakEvenKm:
    def test_zero_cost_per_km_returns_inf(self) -> None:
        assert break_even_km(400.0, 1000.0, 0.0) == float("inf")

    def test_break_even_calculated(self) -> None:
        # saving = 600, cost_per_km = 0.1 → 6000 km
        assert break_even_km(400.0, 1000.0, 0.1) == pytest.approx(6000.0)
