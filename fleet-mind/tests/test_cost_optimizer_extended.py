"""Extended cost optimizer tests with parametrize and boundary coverage."""
from __future__ import annotations

import math

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


def _option(cost: float = 500.0, downtime: float = 4.0, gain: float = 0.1) -> RepairOption:
    return RepairOption(
        description="Test repair",
        cost=cost,
        downtime_hours=downtime,
        reliability_gain=gain,
    )


def _model(labour: float = 85.0, downtime_cost: float = 200.0) -> CostModel:
    return CostModel(labour_rate_per_hour=labour, downtime_cost_per_hour=downtime_cost)


def test_total_repair_cost_includes_downtime() -> None:
    option = _option(cost=100.0, downtime=2.0)
    model = _model(downtime_cost=200.0)
    assert total_repair_cost(option, model) == pytest.approx(100.0 + 2.0 * 200.0)


@pytest.mark.parametrize("cost,downtime,expected", [
    (0.0, 0.0, 0.0),
    (500.0, 0.0, 500.0),
    (0.0, 4.0, 800.0),
    (100.0, 1.0, 300.0),
])
def test_total_repair_cost_parametrized(cost: float, downtime: float, expected: float) -> None:
    option = _option(cost=cost, downtime=downtime)
    model = _model(downtime_cost=200.0)
    assert total_repair_cost(option, model) == pytest.approx(expected)


def test_preventive_vs_reactive_saving_positive_case() -> None:
    assert preventive_vs_reactive_saving(1000.0, 400.0) == pytest.approx(600.0)


def test_preventive_vs_reactive_saving_no_saving_when_preventive_costs_more() -> None:
    assert preventive_vs_reactive_saving(300.0, 500.0) == pytest.approx(0.0)


def test_preventive_vs_reactive_saving_equal_costs() -> None:
    assert preventive_vs_reactive_saving(500.0, 500.0) == pytest.approx(0.0)


def test_cost_per_reliability_unit_zero_gain_returns_inf() -> None:
    option = _option(gain=0.0)
    model = _model()
    result = cost_per_reliability_unit(option, model)
    assert math.isinf(result)


def test_cost_per_reliability_unit_negative_gain_returns_inf() -> None:
    option = _option(gain=-0.1)
    model = _model()
    result = cost_per_reliability_unit(option, model)
    assert math.isinf(result)


@pytest.mark.parametrize("gain", [0.1, 0.5, 1.0])
def test_cost_per_reliability_unit_decreases_with_gain(gain: float) -> None:
    option = _option(cost=100.0, downtime=0.0, gain=gain)
    model = _model(downtime_cost=0.0)
    result = cost_per_reliability_unit(option, model)
    assert result == pytest.approx(100.0 / gain)


def test_rank_options_ascending_by_cost_efficiency() -> None:
    options = [
        _option(cost=1000.0, downtime=0.0, gain=0.1),
        _option(cost=100.0, downtime=0.0, gain=0.5),
        _option(cost=500.0, downtime=0.0, gain=0.3),
    ]
    model = _model(downtime_cost=0.0)
    ranked = rank_options(options, model)
    costs_per_unit = [cost_per_reliability_unit(o, model) for o in ranked]
    assert costs_per_unit == sorted(costs_per_unit)


def test_rank_options_empty_list() -> None:
    assert rank_options([], _model()) == []


def test_rank_options_single_option() -> None:
    options = [_option()]
    assert rank_options(options, _model()) == options


@pytest.mark.parametrize("vehicles,avg_cost,contingency,expected", [
    (10, 1000.0, 0.10, 11000.0),
    (100, 500.0, 0.0, 50000.0),
    (1, 2000.0, 0.20, 2400.0),
])
def test_annual_maintenance_budget(
    vehicles: int, avg_cost: float, contingency: float, expected: float
) -> None:
    result = annual_maintenance_budget(vehicles, avg_cost, contingency_pct=contingency)
    assert result == pytest.approx(expected)


def test_break_even_km_zero_cost_per_km_returns_inf() -> None:
    result = break_even_km(400.0, 1000.0, 0.0)
    assert math.isinf(result)


def test_break_even_km_no_saving_returns_zero() -> None:
    result = break_even_km(1000.0, 500.0, 1.0)
    assert result == pytest.approx(0.0)


@pytest.mark.parametrize("reactive,preventive,cost_per_km,expected_km", [
    (1000.0, 400.0, 0.1, 6000.0),
    (500.0, 100.0, 0.5, 800.0),
    (200.0, 50.0, 0.3, 500.0),
])
def test_break_even_km_various(
    reactive: float, preventive: float, cost_per_km: float, expected_km: float
) -> None:
    result = break_even_km(preventive, reactive, cost_per_km)
    assert result == pytest.approx(expected_km, rel=1e-5)
