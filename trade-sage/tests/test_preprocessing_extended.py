"""Extended tests for preprocessing utilities including new log-return and date features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.preprocessing import add_date_features, compute_log_returns


class TestComputeLogReturns:
    def test_empty_series_returns_empty(self) -> None:
        result = compute_log_returns(np.array([]))
        assert len(result) == 0

    def test_single_price_returns_empty(self) -> None:
        result = compute_log_returns(np.array([100.0]))
        assert len(result) == 0

    def test_flat_prices_return_near_zero(self) -> None:
        prices = np.full(10, 100.0)
        result = compute_log_returns(prices)
        assert result.shape == (9,)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_doubling_price_gives_log2(self) -> None:
        prices = np.array([100.0, 200.0])
        result = compute_log_returns(prices)
        assert abs(result[0] - np.log(2)) < 1e-6

    def test_output_length_is_n_minus_one(self) -> None:
        rng = np.random.default_rng(0)
        prices = 100 + rng.standard_normal(50).cumsum()
        result = compute_log_returns(prices)
        assert len(result) == 49

    @pytest.mark.parametrize("n", [2, 5, 20, 100])
    def test_length_parametrized(self, n: int) -> None:
        prices = np.linspace(100, 200, n)
        result = compute_log_returns(prices)
        assert len(result) == n - 1

    def test_list_input_accepted(self) -> None:
        result = compute_log_returns([100.0, 110.0, 105.0])
        assert len(result) == 2


class TestAddDateFeatures:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({"date": ["2024-01-01", "2024-06-15", "2024-12-31"]})

    def test_day_of_week_column_added(self) -> None:
        df = self._make_df()
        result = add_date_features(df)
        assert "day_of_week" in result.columns

    def test_month_column_added(self) -> None:
        df = self._make_df()
        result = add_date_features(df)
        assert "month" in result.columns

    def test_day_of_week_range(self) -> None:
        df = self._make_df()
        result = add_date_features(df)
        assert result["day_of_week"].between(0, 6).all()

    def test_month_range(self) -> None:
        df = self._make_df()
        result = add_date_features(df)
        assert result["month"].between(1, 12).all()

    def test_original_date_col_preserved(self) -> None:
        df = self._make_df()
        result = add_date_features(df)
        assert "date" in result.columns

    def test_june_15_day_of_week(self) -> None:
        df = pd.DataFrame({"date": ["2024-06-15"]})
        result = add_date_features(df)
        assert result["day_of_week"].iloc[0] == 5  # Saturday

    def test_june_month_value(self) -> None:
        df = pd.DataFrame({"date": ["2024-06-15"]})
        result = add_date_features(df)
        assert result["month"].iloc[0] == 6
