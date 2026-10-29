"""Tests for signal-driven backtester."""

from __future__ import annotations

import numpy as np
import pytest

from app.backtest import run_backtest


def _prices(n: int = 100, trend: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(42)
    returns = rng.normal(trend, 0.01, n - 1)
    prices = np.concatenate([[100.0], 100.0 * np.cumprod(1 + returns)])
    return prices


def test_all_buy_signals_positive_trend():
    prices = _prices(100, trend=0.005)
    signals = np.ones(100, dtype=int)
    result = run_backtest(prices, signals)
    assert result.total_return > 0


def test_all_flat_signals_zero_return():
    prices = _prices(100)
    signals = np.zeros(100, dtype=int)
    result = run_backtest(prices, signals)
    assert abs(result.total_return) < 1e-6


def test_equity_curve_length():
    prices = _prices(50)
    signals = np.ones(50, dtype=int)
    result = run_backtest(prices, signals)
    assert len(result.equity_curve) == 50


def test_max_drawdown_non_positive():
    prices = _prices(100)
    signals = np.ones(100, dtype=int)
    result = run_backtest(prices, signals)
    assert result.max_drawdown <= 0.0


def test_win_rate_in_range():
    prices = _prices(100)
    signals = np.ones(100, dtype=int)
    result = run_backtest(prices, signals)
    assert 0.0 <= result.win_rate <= 1.0


def test_mismatched_lengths_raises():
    prices = _prices(10)
    signals = np.ones(8, dtype=int)
    with pytest.raises(ValueError, match="same length"):
        run_backtest(prices, signals)


def test_short_series_returns_zero():
    prices = np.array([100.0])
    signals = np.array([1])
    result = run_backtest(prices, signals)
    assert result.total_return == 0.0


def test_result_dataclass_fields():
    prices = _prices(60)
    signals = np.ones(60, dtype=int)
    result = run_backtest(prices, signals)
    assert hasattr(result, "total_return")
    assert hasattr(result, "annualised_return")
    assert hasattr(result, "max_drawdown")
    assert hasattr(result, "n_trades")
