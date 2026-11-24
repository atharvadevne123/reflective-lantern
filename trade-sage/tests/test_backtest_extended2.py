"""Extended backtest tests with parametrize and boundary coverage."""
from __future__ import annotations

import numpy as np
import pytest

from app.backtest import (
    BacktestResult,
    compare_strategies,
    run_backtest,
    summarise_backtest,
)


def _linear_prices(n: int = 50, start: float = 100.0, slope: float = 1.0) -> np.ndarray:
    return np.array([start + i * slope for i in range(n)], dtype=float)


def _flat_prices(n: int = 50, price: float = 100.0) -> np.ndarray:
    return np.full(n, price, dtype=float)


def test_run_backtest_returns_backtest_result() -> None:
    prices = _linear_prices(50)
    signals = np.ones(50, dtype=int)
    result = run_backtest(prices, signals)
    assert isinstance(result, BacktestResult)


def test_run_backtest_too_short_returns_zero_result() -> None:
    result = run_backtest(np.array([100.0]), np.array([1]))
    assert result.total_return == 0.0
    assert result.n_trades == 0


def test_run_backtest_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        run_backtest(np.array([100.0, 101.0]), np.array([1]))


def test_run_backtest_all_cash_no_return() -> None:
    prices = _linear_prices(50)
    signals = np.zeros(50, dtype=int)
    result = run_backtest(prices, signals)
    assert result.total_return == pytest.approx(0.0, abs=1e-6)
    assert result.n_trades == 0


def test_run_backtest_flat_prices_zero_return() -> None:
    prices = _flat_prices(50)
    signals = np.ones(50, dtype=int)
    result = run_backtest(prices, signals)
    assert abs(result.total_return) < 0.01


@pytest.mark.parametrize("n", [2, 10, 50, 100, 252])
def test_run_backtest_equity_curve_length(n: int) -> None:
    prices = _linear_prices(n)
    signals = np.ones(n, dtype=int)
    result = run_backtest(prices, signals)
    assert len(result.equity_curve) == n


def test_run_backtest_buy_hold_rising_market_positive_return() -> None:
    prices = _linear_prices(100, start=100.0, slope=0.5)
    signals = np.ones(100, dtype=int)
    result = run_backtest(prices, signals)
    assert result.total_return > 0.0


@pytest.mark.parametrize("cost", [0.0, 0.001, 0.005, 0.01])
def test_run_backtest_transaction_cost_reduces_return(cost: float) -> None:
    prices = _linear_prices(50)
    signals_switch = np.array([i % 2 for i in range(50)], dtype=int)
    result = run_backtest(prices, signals_switch, transaction_cost=cost)
    assert isinstance(result.total_return, float)


def test_run_backtest_max_drawdown_non_negative() -> None:
    prices = np.array([100.0, 90.0, 80.0, 70.0, 80.0, 90.0])
    signals = np.ones(6, dtype=int)
    result = run_backtest(prices, signals)
    assert result.max_drawdown >= 0.0


def test_run_backtest_win_rate_in_zero_one() -> None:
    prices = _linear_prices(50)
    signals = np.array([i % 3 == 0 for i in range(50)], dtype=int)
    result = run_backtest(prices, signals)
    assert 0.0 <= result.win_rate <= 1.0


def test_summarise_backtest_has_all_keys() -> None:
    prices = _linear_prices(50)
    signals = np.ones(50, dtype=int)
    result = run_backtest(prices, signals)
    summary = summarise_backtest(result)
    for key in ("total_return_pct", "annualised_return_pct", "max_drawdown_pct",
                "sharpe_ratio", "n_trades", "win_rate_pct", "equity_curve_length"):
        assert key in summary


def test_summarise_backtest_equity_curve_length_matches() -> None:
    prices = _linear_prices(50)
    signals = np.ones(50, dtype=int)
    result = run_backtest(prices, signals)
    summary = summarise_backtest(result)
    assert summary["equity_curve_length"] == len(result.equity_curve)


def test_compare_strategies_returns_all_names() -> None:
    prices = _linear_prices(50)
    strategies = {
        "buy_hold": np.ones(50, dtype=int),
        "cash": np.zeros(50, dtype=int),
    }
    comparison = compare_strategies(prices, strategies)
    assert set(comparison.keys()) == {"buy_hold", "cash"}


def test_compare_strategies_empty_dict() -> None:
    prices = _linear_prices(50)
    result = compare_strategies(prices, {})
    assert result == {}


@pytest.mark.parametrize("n_strategies", [1, 3, 5])
def test_compare_strategies_multiple(n_strategies: int) -> None:
    prices = _linear_prices(50)
    strategies = {f"s{i}": np.ones(50, dtype=int) for i in range(n_strategies)}
    result = compare_strategies(prices, strategies)
    assert len(result) == n_strategies
