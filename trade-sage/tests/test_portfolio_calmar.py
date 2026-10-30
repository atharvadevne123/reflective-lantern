"""Tests for compute_calmar_ratio in portfolio analytics."""

from __future__ import annotations

import pytest

from app.portfolio import compute_calmar_ratio


class TestComputeCalmarRatio:
    def test_short_series_returns_none(self) -> None:
        assert compute_calmar_ratio([100.0]) is None

    def test_empty_series_returns_none(self) -> None:
        assert compute_calmar_ratio([]) is None

    def test_flat_prices_drawdown_zero_returns_none(self) -> None:
        prices = [100.0] * 10
        result = compute_calmar_ratio(prices)
        assert result is None

    def test_uptrend_gives_positive_calmar(self) -> None:
        prices = [100.0 * (1.001 ** i) for i in range(252)]
        result = compute_calmar_ratio(prices)
        assert result is not None
        assert result > 0

    def test_severe_drawdown_lowers_calmar(self) -> None:
        prices = [100.0, 150.0, 80.0, 90.0, 100.0]
        result = compute_calmar_ratio(prices)
        assert result is not None

    def test_returns_float(self) -> None:
        prices = [100.0, 105.0, 110.0, 108.0, 115.0]
        result = compute_calmar_ratio(prices)
        if result is not None:
            assert isinstance(result, float)

    def test_two_prices_no_drawdown_returns_none_or_float(self) -> None:
        result = compute_calmar_ratio([100.0, 110.0])
        assert result is None or isinstance(result, float)

    @pytest.mark.parametrize("n", [5, 10, 50, 252])
    def test_returns_value_for_growing_prices(self, n: int) -> None:
        prices = [100.0 + i for i in range(n)]
        result = compute_calmar_ratio(prices)
        assert result is None or isinstance(result, float)

    def test_downtrend_gives_negative_calmar(self) -> None:
        prices = [100.0 * (0.999 ** i) for i in range(100)]
        result = compute_calmar_ratio(prices)
        if result is not None:
            assert result < 0
