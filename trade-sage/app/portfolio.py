"""Portfolio risk analytics: VaR, ES, volatility, Sharpe, max drawdown."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_returns(close_prices: list[float]) -> np.ndarray:
    arr = np.array(close_prices, dtype=float)
    return np.diff(arr) / (arr[:-1] + 1e-9)


def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (daily, negative = loss)."""
    if len(returns) < 5:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def compute_expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall: mean of tail losses beyond VaR."""
    if len(returns) < 5:
        return 0.0
    var = compute_var(returns, confidence)
    tail = returns[returns <= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


def compute_max_drawdown(close_prices: list[float]) -> float:
    arr = np.array(close_prices, dtype=float)
    if len(arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / (peak + 1e-9)
    return float(drawdown.min())


def compute_sharpe(returns: np.ndarray, risk_free_rate_daily: float = 0.0) -> float | None:
    if len(returns) < 5:
        return None
    excess = returns - risk_free_rate_daily
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return None
    # annualise assuming 252 trading days
    return float((np.mean(excess) / std) * np.sqrt(252))


def compute_portfolio_risk(
    weights: list[float],
    price_series: list[list[float]],
    confidence: float = 0.95,
) -> dict:
    """Aggregate portfolio risk given weights and per-asset close series."""
    returns_matrix = []
    for prices in price_series:
        r = compute_returns(prices)
        returns_matrix.append(r)

    min_len = min(len(r) for r in returns_matrix)
    ret_arr = np.column_stack([r[-min_len:] for r in returns_matrix])
    w = np.array(weights, dtype=float)

    portfolio_returns = ret_arr @ w
    port_vol = float(np.std(portfolio_returns, ddof=1) * np.sqrt(252))

    if len(weights) > 1:
        corr_matrix = np.corrcoef(ret_arr.T)
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        correlation_avg = float(np.mean(np.abs(corr_matrix[mask])))
    else:
        correlation_avg = 1.0

    all_prices = np.column_stack([np.array(p[-min_len - 1:]) for p in price_series])
    portfolio_value = all_prices @ w
    mdd = compute_max_drawdown(portfolio_value.tolist())

    return {
        "value_at_risk": compute_var(portfolio_returns, confidence),
        "expected_shortfall": compute_expected_shortfall(portfolio_returns, confidence),
        "portfolio_volatility": port_vol,
        "sharpe_ratio": compute_sharpe(portfolio_returns),
        "max_drawdown": mdd,
        "correlation_avg": correlation_avg,
    }
