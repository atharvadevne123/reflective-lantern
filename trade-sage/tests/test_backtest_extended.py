"""Extended tests for backtest summarise and compare utilities."""

from __future__ import annotations

import numpy as np
import pytest

from app.backtest import compare_strategies, run_backtest, summarise_backtest


def _prices(n: int = 100) -> np.ndarray:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, n - 1)
    return np.concatenate([[100.0], 100.0 * np.cumprod(1 + returns)])


def _all_buy(n: int = 100) -> np.ndarray:
    return np.ones(n, dtype=int)


def _all_sell(n: int = 100) -> np.ndarray:
    return np.zeros(n, dtype=int)


class TestSummariseBacktest:
    def test_returns_dict(self) -> None:
        result = run_backtest(_prices(), _all_buy())
        summary = summarise_backtest(result)
        assert isinstance(summary, dict)

    def test_has_expected_keys(self) -> None:
        result = run_backtest(_prices(), _all_buy())
        summary = summarise_backtest(result)
        expected = {"total_return_pct", "annualised_return_pct", "max_drawdown_pct",
                    "sharpe_ratio", "n_trades", "win_rate_pct", "equity_curve_length"}
        assert expected == set(summary.keys())

    def test_returns_in_percentage(self) -> None:
        result = run_backtest(_prices(100), _all_sell())
        summary = summarise_backtest(result)
        assert abs(summary["total_return_pct"]) < 100

    def test_equity_curve_length_matches(self) -> None:
        prices = _prices(50)
        result = run_backtest(prices, _all_buy(50))
        summary = summarise_backtest(result)
        assert summary["equity_curve_length"] == len(result.equity_curve)

    def test_max_drawdown_pct_negative_or_zero(self) -> None:
        result = run_backtest(_prices(), _all_buy())
        summary = summarise_backtest(result)
        assert summary["max_drawdown_pct"] <= 0.0

    def test_win_rate_pct_in_range(self) -> None:
        result = run_backtest(_prices(), _all_buy())
        summary = summarise_backtest(result)
        assert 0.0 <= summary["win_rate_pct"] <= 100.0


class TestCompareStrategies:
    def test_returns_dict(self) -> None:
        prices = _prices()
        signals = {"buy_all": _all_buy(), "hold_cash": _all_sell()}
        result = compare_strategies(prices, signals)
        assert isinstance(result, dict)

    def test_has_all_strategy_names(self) -> None:
        prices = _prices()
        strategies = {"strat_a": _all_buy(), "strat_b": _all_sell()}
        result = compare_strategies(prices, strategies)
        assert "strat_a" in result
        assert "strat_b" in result

    def test_single_strategy(self) -> None:
        prices = _prices()
        result = compare_strategies(prices, {"only": _all_buy()})
        assert len(result) == 1

    def test_each_result_has_total_return(self) -> None:
        prices = _prices()
        result = compare_strategies(prices, {"a": _all_buy(), "b": _all_sell()})
        for _, summary in result.items():
            assert "total_return_pct" in summary

    @pytest.mark.parametrize("cost", [0.0, 0.001, 0.005])
    def test_transaction_cost_parametrized(self, cost: float) -> None:
        prices = _prices()
        result = compare_strategies(prices, {"s": _all_buy()}, transaction_cost=cost)
        assert isinstance(result["s"]["total_return_pct"], float)
