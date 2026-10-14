"""Simple signal-driven backtester for equity trading strategies."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BacktestResult:
    """Aggregated results from a signal-driven backtest.

    Attributes:
        total_return: Cumulative strategy return over the period.
        annualised_return: Annualised equivalent of total_return.
        max_drawdown: Maximum peak-to-trough decline.
        sharpe_ratio: Annualised Sharpe ratio (risk-free rate = 0).
        n_trades: Number of position changes triggered.
        win_rate: Fraction of trades with positive PnL.
    """

    total_return: float
    annualised_return: float
    max_drawdown: float
    sharpe_ratio: float | None
    n_trades: int
    win_rate: float
    equity_curve: list[float] = field(default_factory=list)


def run_backtest(
    close_prices: np.ndarray,
    signals: np.ndarray,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Run a long-only signal backtest on a sequence of close prices.

    Args:
        close_prices: 1D array of daily close prices (length N).
        signals: 1D int array of {0, 1} buy/hold signals, same length as prices.
                 1 = hold long, 0 = flat/cash.
        transaction_cost: Round-trip cost fraction applied on each trade entry/exit.

    Returns:
        BacktestResult with performance metrics.
    """
    if len(close_prices) != len(signals):
        raise ValueError("close_prices and signals must have the same length")
    if len(close_prices) < 2:
        return BacktestResult(0.0, 0.0, 0.0, None, 0, 0.0)

    daily_returns = np.diff(close_prices) / close_prices[:-1]
    positions = signals[:-1].astype(float)

    trades = np.abs(np.diff(np.concatenate([[0], positions])))
    strategy_returns = positions * daily_returns - trades * transaction_cost

    equity = np.cumprod(1 + strategy_returns)
    equity_curve = [1.0, *equity.tolist()]

    total_return = float(equity[-1] - 1.0)
    n_days = len(daily_returns)
    trading_days = 252
    annualised_return = float((1 + total_return) ** (trading_days / n_days) - 1) if n_days > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(drawdown.min())

    if strategy_returns.std() > 0:
        sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(trading_days))
    else:
        sharpe = None

    trade_indices = np.where(trades[1:] > 0)[0]
    n_trades = int(trades.sum())
    if len(trade_indices) > 0:
        trade_pnl = strategy_returns[trade_indices]
        win_rate = float((trade_pnl > 0).mean())
    else:
        win_rate = 0.0

    return BacktestResult(
        total_return=round(total_return, 6),
        annualised_return=round(annualised_return, 6),
        max_drawdown=round(max_drawdown, 6),
        sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
        n_trades=n_trades,
        win_rate=round(win_rate, 4),
        equity_curve=equity_curve,
    )
