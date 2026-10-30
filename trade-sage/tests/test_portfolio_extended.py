"""Extended tests for trade-sage app/portfolio.py risk analytics."""

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


class TestComputeReturns:
    def test_basic_returns(self) -> None:
        prices = [100.0, 110.0, 99.0]
        returns = compute_returns(prices)
        assert len(returns) == 2

    def test_flat_prices_near_zero_returns(self) -> None:
        prices = [100.0] * 5
        returns = compute_returns(prices)
        assert all(abs(r) < 1e-6 for r in returns)

    def test_increasing_prices_positive_returns(self) -> None:
        prices = [10.0, 11.0, 12.0, 13.0]
        returns = compute_returns(prices)
        assert all(r > 0 for r in returns)


class TestComputeVaR:
    def test_returns_float(self) -> None:
        returns = np.random.randn(100) * 0.02
        var = compute_var(returns)
        assert isinstance(var, float)

    def test_var_is_negative_for_losses(self) -> None:
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03] * 10)
        var = compute_var(returns, confidence=0.95)
        assert var < 0

    def test_too_few_returns_zero(self) -> None:
        returns = np.array([0.01, 0.02])
        assert compute_var(returns) == pytest.approx(0.0)

    def test_higher_confidence_lower_var(self) -> None:
        returns = np.random.randn(200) * 0.02
        var95 = compute_var(returns, confidence=0.95)
        var99 = compute_var(returns, confidence=0.99)
        assert var99 <= var95


class TestComputeExpectedShortfall:
    def test_returns_float(self) -> None:
        returns = np.random.randn(100) * 0.02
        es = compute_expected_shortfall(returns)
        assert isinstance(es, float)

    def test_es_le_var(self) -> None:
        returns = np.random.randn(200) * 0.02
        var = compute_var(returns, confidence=0.95)
        es = compute_expected_shortfall(returns, confidence=0.95)
        assert es <= var

    def test_too_few_returns_zero(self) -> None:
        returns = np.array([0.01, 0.02])
        assert compute_expected_shortfall(returns) == pytest.approx(0.0)


class TestComputeMaxDrawdown:
    def test_monotonically_increasing_no_drawdown(self) -> None:
        prices = [100.0, 110.0, 120.0, 130.0]
        dd = compute_max_drawdown(prices)
        assert dd == pytest.approx(0.0, abs=1e-4)

    def test_drawdown_is_negative(self) -> None:
        prices = [100.0, 80.0, 90.0, 70.0, 110.0]
        dd = compute_max_drawdown(prices)
        assert dd < 0

    def test_single_price_no_drawdown(self) -> None:
        assert compute_max_drawdown([100.0]) == pytest.approx(0.0)


class TestComputeSharpe:
    def test_returns_float_or_none(self) -> None:
        returns = np.random.randn(100) * 0.01
        result = compute_sharpe(returns)
        assert result is None or isinstance(result, float)

    def test_constant_returns_none(self) -> None:
        returns = np.full(100, 0.001)
        result = compute_sharpe(returns)
        assert result is None

    def test_too_few_returns_none(self) -> None:
        assert compute_sharpe(np.array([0.01, 0.02])) is None
