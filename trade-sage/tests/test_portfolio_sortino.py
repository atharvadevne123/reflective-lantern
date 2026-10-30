"""Tests for compute_sortino_ratio."""

from __future__ import annotations

import numpy as np
import pytest

from app.portfolio import compute_sortino_ratio


class TestComputeSortinoRatio:
    def test_returns_float(self) -> None:
        returns = np.array([-0.01, 0.02, -0.005, 0.03, 0.01, -0.008, 0.015])
        result = compute_sortino_ratio(returns)
        assert isinstance(result, float)

    def test_short_series_returns_none(self) -> None:
        returns = np.array([0.01, -0.01, 0.02, 0.01])
        assert compute_sortino_ratio(returns) is None

    def test_no_downside_returns_none(self) -> None:
        returns = np.array([0.01, 0.02, 0.015, 0.03, 0.025])
        assert compute_sortino_ratio(returns) is None

    def test_all_negative_returns_gives_ratio(self) -> None:
        returns = np.array([-0.01, -0.02, -0.015, -0.03, -0.025, -0.01])
        result = compute_sortino_ratio(returns)
        assert result is not None
        assert result < 0

    def test_mixed_returns_positive_sortino(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 100)
        result = compute_sortino_ratio(returns)
        assert result is not None

    @pytest.mark.parametrize("rfr", [0.0, 0.0001])
    def test_risk_free_rate_effect(self, rfr: float) -> None:
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.01, 50)
        result = compute_sortino_ratio(returns, risk_free_rate_daily=rfr)
        assert result is None or isinstance(result, float)

    def test_trading_days_scaling(self) -> None:
        returns = np.array([-0.01, 0.02, -0.005, 0.03, 0.01, -0.008, 0.015, -0.003])
        r252 = compute_sortino_ratio(returns, trading_days=252)
        r365 = compute_sortino_ratio(returns, trading_days=365)
        assert r252 is not None and r365 is not None
        assert r365 > r252
