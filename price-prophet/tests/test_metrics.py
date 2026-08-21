"""Tests for app/evaluation/metrics.py."""

from __future__ import annotations

import math

import pytest


def test_mae_perfect():
    from app.evaluation.metrics import mae

    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_basic():
    from app.evaluation.metrics import mae

    result = mae([0.0, 10.0], [5.0, 5.0])
    assert abs(result - 5.0) < 1e-9


def test_mae_empty():
    from app.evaluation.metrics import mae

    with pytest.raises(ValueError):
        mae([], [])


def test_rmse_basic():
    from app.evaluation.metrics import rmse

    result = rmse([0.0, 0.0, 0.0], [3.0, 4.0, 5.0])
    expected = math.sqrt((9 + 16 + 25) / 3)
    assert abs(result - expected) < 1e-9


def test_mape_basic():
    from app.evaluation.metrics import mape

    result = mape([100.0, 200.0], [90.0, 220.0])
    assert result > 0


def test_r_squared_perfect():
    from app.evaluation.metrics import r_squared

    assert r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_revenue_uplift_basic():
    from app.evaluation.metrics import revenue_uplift

    uplift = revenue_uplift(100.0, 110.0)
    assert abs(uplift - 10.0) < 1e-9


def test_revenue_uplift_zero_baseline():
    from app.evaluation.metrics import revenue_uplift

    assert revenue_uplift(0.0, 100.0) == 0.0


@pytest.mark.parametrize(
    "actual,predicted,expected_mae",
    [
        ([1.0], [1.0], 0.0),
        ([0.0, 10.0, 20.0], [0.0, 10.0, 20.0], 0.0),
        ([0.0, 0.0], [5.0, 5.0], 5.0),
        ([10.0], [0.0], 10.0),
    ],
)
def test_mae_parametrized(actual: list[float], predicted: list[float], expected_mae: float) -> None:
    from app.evaluation.metrics import mae

    assert abs(mae(actual, predicted) - expected_mae) < 1e-6


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ([5.0, 5.0], [5.0, 5.0]),
    ],
)
def test_r_squared_perfect_fit_parametrized(actual: list[float], predicted: list[float]) -> None:
    from app.evaluation.metrics import r_squared

    assert r_squared(actual, predicted) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("baseline,optimized", [(100.0, 110.0), (50.0, 75.0), (200.0, 200.0)])
def test_revenue_uplift_non_negative_when_optimized_ge_baseline(
    baseline: float, optimized: float
) -> None:
    from app.evaluation.metrics import revenue_uplift

    result = revenue_uplift(baseline, optimized)
    assert result >= 0.0


def test_mae_empty_raises() -> None:
    import pytest

    from app.evaluation.metrics import mae

    with pytest.raises(ValueError):
        mae([], [])


def test_mape_all_zero_actuals_returns_zero() -> None:
    from app.evaluation.metrics import mape

    assert mape([0.0, 0.0], [5.0, 10.0]) == 0.0


def test_revenue_uplift_zero_baseline_returns_zero() -> None:
    from app.evaluation.metrics import revenue_uplift

    assert revenue_uplift(0.0, 100.0) == 0.0


def test_rmse_single_element() -> None:
    from app.evaluation.metrics import rmse

    assert rmse([3.0], [1.0]) == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("v", [10.0, 100.0, 1000.0])
def test_r_squared_worst_fit_negative(v: float) -> None:
    from app.evaluation.metrics import r_squared

    # Predicting the opposite direction should give negative R2
    actual = [0.0, v]
    predicted = [v, 0.0]
    assert r_squared(actual, predicted) < 0.0
