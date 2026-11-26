"""Extended portfolio risk analytics tests — second pass with parametrize."""

from __future__ import annotations

import numpy as np
import pytest

from app.portfolio import (
    compute_expected_shortfall,
    compute_max_drawdown,
    compute_returns,
    compute_sharpe,
    compute_var,
)


def _rising_prices(n: int = 10) -> list[float]:
    return [100.0 + i for i in range(n)]


def _falling_prices(n: int = 10) -> list[float]:
    return [100.0 - i for i in range(n)]


def test_compute_returns_length_one_less_than_prices() -> None:
    prices = _rising_prices(10)
    returns = compute_returns(prices)
    assert len(returns) == 9


def test_compute_returns_rising_all_positive() -> None:
    prices = _rising_prices(10)
    returns = compute_returns(prices)
    assert all(r > 0 for r in returns)


def test_compute_returns_falling_all_negative() -> None:
    prices = _falling_prices(10)
    returns = compute_returns(prices)
    assert all(r < 0 for r in returns)


def test_compute_var_too_few_returns_zero() -> None:
    returns = np.array([0.01, 0.02, 0.03])
    assert compute_var(returns) == pytest.approx(0.0)


def test_compute_var_negative_tail() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, 200)
    var = compute_var(returns, confidence=0.95)
    assert var < 0


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_compute_var_stricter_confidence_lower_var(confidence: float) -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, 200)
    var = compute_var(returns, confidence=confidence)
    assert isinstance(var, float)


def test_compute_expected_shortfall_leq_var() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, 200)
    var = compute_var(returns)
    es = compute_expected_shortfall(returns)
    assert es <= var


def test_compute_expected_shortfall_too_few_returns_zero() -> None:
    returns = np.array([0.01, 0.02])
    assert compute_expected_shortfall(returns) == pytest.approx(0.0)


def test_compute_max_drawdown_flat_prices_zero() -> None:
    prices = [100.0] * 10
    assert compute_max_drawdown(prices) == pytest.approx(0.0, abs=1e-6)


def test_compute_max_drawdown_falling_prices_negative() -> None:
    prices = [100.0, 90.0, 80.0, 70.0]
    mdd = compute_max_drawdown(prices)
    assert mdd < 0


def test_compute_max_drawdown_too_few_prices_zero() -> None:
    assert compute_max_drawdown([100.0]) == pytest.approx(0.0)


def test_compute_max_drawdown_v_shape_recovery() -> None:
    prices = [100.0, 50.0, 100.0]
    mdd = compute_max_drawdown(prices)
    assert mdd < 0


def test_compute_sharpe_too_few_returns_none() -> None:
    returns = np.array([0.01, 0.02, 0.03])
    assert compute_sharpe(returns) is None


def test_compute_sharpe_positive_returns_positive_sharpe() -> None:
    rng = np.random.default_rng(1)
    returns = np.abs(rng.normal(0.001, 0.005, 200))
    sharpe = compute_sharpe(returns)
    assert sharpe is not None
    assert sharpe > 0


def test_compute_sharpe_zero_std_returns_none() -> None:
    returns = np.ones(50) * 0.01
    assert compute_sharpe(returns) is None


@pytest.mark.parametrize("n_prices", [2, 10, 50, 100])
def test_compute_returns_various_lengths(n_prices: int) -> None:
    prices = [100.0 + 0.1 * i for i in range(n_prices)]
    returns = compute_returns(prices)
    assert len(returns) == n_prices - 1
