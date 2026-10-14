"""Tests for portfolio risk analytics."""
from __future__ import annotations

import numpy as np
import pytest

from app.portfolio import (
    compute_expected_shortfall,
    compute_max_drawdown,
    compute_portfolio_risk,
    compute_returns,
    compute_sharpe,
    compute_var,
)


def _price_series(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return (100 + np.cumsum(rng.normal(0, 1, n))).tolist()


def test_compute_returns_length():
    prices = _price_series(50)
    returns = compute_returns(prices)
    assert len(returns) == 49


def test_compute_returns_values():
    prices = [100.0, 105.0, 100.0]
    r = compute_returns(prices)
    assert abs(r[0] - 0.05) < 0.001
    assert abs(r[1] - (-5 / 105)) < 0.001


def test_var_negative_for_normal():
    rng = np.random.default_rng(5)
    r = rng.normal(-0.001, 0.02, 1000)
    var = compute_var(r, 0.95)
    assert var < 0


def test_var_insufficient_data():
    assert compute_var(np.array([1.0, 2.0, 3.0]), 0.95) == 0.0


def test_expected_shortfall_lte_var():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.01, 500)
    var = compute_var(r, 0.95)
    es = compute_expected_shortfall(r, 0.95)
    assert es <= var


def test_max_drawdown_negative():
    prices = [100.0, 110.0, 90.0, 80.0, 95.0]
    mdd = compute_max_drawdown(prices)
    assert mdd < 0


def test_max_drawdown_monotone_up():
    prices = [100.0, 110.0, 120.0, 130.0]
    mdd = compute_max_drawdown(prices)
    assert mdd == pytest.approx(0.0, abs=0.01)


def test_sharpe_positive_trend():
    # Use a deterministic series with clear positive mean rather than relying on RNG
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.001, 252)
    r = np.full(252, 0.005) + noise  # 0.5% daily drift, tiny noise
    sharpe = compute_sharpe(r)
    assert sharpe is not None
    assert sharpe > 0


def test_sharpe_none_for_zero_vol():
    sharpe = compute_sharpe(np.zeros(100))
    assert sharpe is None


def test_compute_portfolio_risk_single_asset():
    prices = _price_series(100)
    result = compute_portfolio_risk([1.0], [prices], 0.95)
    assert "value_at_risk" in result
    assert "portfolio_volatility" in result
    assert result["portfolio_volatility"] >= 0


def test_compute_portfolio_risk_two_assets():
    p1 = _price_series(100, 0)
    p2 = _price_series(100, 1)
    result = compute_portfolio_risk([0.6, 0.4], [p1, p2], 0.95)
    assert "correlation_avg" in result
    assert 0 <= result["correlation_avg"] <= 1


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_var_increases_with_confidence(confidence):
    rng = np.random.default_rng(8)
    r = rng.normal(0, 0.01, 1000)
    var90 = compute_var(r, 0.90)
    var_c = compute_var(r, confidence)
    if confidence > 0.90:
        assert var_c <= var90  # higher confidence = more negative VaR
