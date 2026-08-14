"""Tests for app/evaluation/backtester.py."""

from __future__ import annotations

import pytest


class _FlatModel:
    def predict(self, X):
        return [10.0] * len(X)


def _make_backtester(baseline=None):
    from app.evaluation.backtester import Backtester

    if baseline is None:
        baseline = [100.0, 100.0, 100.0]
    return Backtester(_FlatModel(), baseline)


def test_backtester_returns_dict():
    bt = _make_backtester()
    result = bt.run([[1.0]] * 3, [10.0, 10.0, 10.0], [110.0, 110.0, 110.0])
    assert isinstance(result, dict)


def test_backtester_n_periods():
    bt = _make_backtester()
    result = bt.run([[1.0]] * 3, [10.0, 10.0, 10.0], [110.0, 110.0, 110.0])
    assert result["n_periods"] == 3


def test_backtester_empty():
    bt = _make_backtester()
    result = bt.run([], [], [])
    assert result["n_periods"] == 0
    assert result["baseline_revenue"] == 0.0


def test_backtester_baseline_revenue():
    bt = _make_backtester([50.0, 50.0])
    result = bt.run([[1.0]] * 2, [10.0, 10.0], [55.0, 55.0])
    assert result["baseline_revenue"] == 1000.0


@pytest.mark.parametrize("n_periods", [1, 3, 5, 10])
def test_backtester_n_periods(n_periods: int) -> None:
    bt = _make_backtester([100.0] * n_periods)
    features = [[1.0]] * n_periods
    demands = [10.0] * n_periods
    actuals = [105.0] * n_periods
    result = bt.run(features, demands, actuals)
    assert result["n_periods"] == n_periods


@pytest.mark.parametrize("baseline_price", [50.0, 100.0, 200.0])
def test_backtester_baseline_revenue_scales_with_price(baseline_price: float) -> None:
    n = 3
    bt = _make_backtester([baseline_price] * n)
    demands = [5.0] * n
    result = bt.run([[1.0]] * n, demands, [baseline_price * 1.1] * n)
    assert result["baseline_revenue"] == pytest.approx(baseline_price * 5.0 * n, abs=0.01)
