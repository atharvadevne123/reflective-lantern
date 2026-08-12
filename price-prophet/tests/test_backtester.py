"""Tests for app/evaluation/backtester.py."""
from __future__ import annotations


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
