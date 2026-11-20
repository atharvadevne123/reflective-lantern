"""Extended tests for signal-forge app/data_utils.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from app.data_utils import (
    compute_log_returns,
    fill_missing_ohlcv,
    validate_ohlcv_frame,
    winsorise,
    zscore_normalise,
)


class TestFillMissingOhlcv:
    def test_fills_nan_with_ffill(self) -> None:
        df = pd.DataFrame({
            "close": [1.0, np.nan, 3.0],
            "volume": [100.0, np.nan, 300.0],
        })
        result = fill_missing_ohlcv(df)
        assert not result["close"].isna().any()

    def test_returns_copy_not_inplace(self) -> None:
        df = pd.DataFrame({"close": [1.0, np.nan]})
        result = fill_missing_ohlcv(df)
        assert result is not df

    def test_non_ohlcv_columns_unchanged(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0], "label": ["a", None]})
        result = fill_missing_ohlcv(df)
        assert result["label"].iloc[1] is None


class TestComputeLogReturns:
    def test_first_element_is_nan(self) -> None:
        prices = pd.Series([100.0, 110.0, 105.0])
        returns = compute_log_returns(prices)
        assert np.isnan(returns.iloc[0])

    def test_length_preserved(self) -> None:
        prices = pd.Series([100.0, 110.0, 105.0, 120.0])
        returns = compute_log_returns(prices)
        assert len(returns) == len(prices)

    def test_positive_return_for_increasing_price(self) -> None:
        prices = pd.Series([100.0, 110.0])
        returns = compute_log_returns(prices)
        assert returns.iloc[1] > 0

    def test_negative_return_for_decreasing_price(self) -> None:
        prices = pd.Series([110.0, 100.0])
        returns = compute_log_returns(prices)
        assert returns.iloc[1] < 0


class TestWinsorise:
    def test_clips_high_outlier(self) -> None:
        series = pd.Series([1.0] * 99 + [1000.0])
        result = winsorise(series)
        assert result.max() < 1000.0

    def test_clips_low_outlier(self) -> None:
        series = pd.Series([-1000.0] + [1.0] * 99)
        result = winsorise(series)
        assert result.min() > -1000.0

    def test_length_preserved(self) -> None:
        series = pd.Series(range(100), dtype=float)
        result = winsorise(series)
        assert len(result) == 100


class TestZscoreNormalise:
    def test_zero_mean_unit_std(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = zscore_normalise(series)
        assert result.mean() == pytest.approx(0.0, abs=1e-10)
        assert result.std() == pytest.approx(1.0, abs=0.01)

    def test_constant_series_returns_zeros(self) -> None:
        series = pd.Series([5.0] * 5)
        result = zscore_normalise(series)
        assert (result == 0.0).all()

    def test_length_preserved(self) -> None:
        series = pd.Series(range(10), dtype=float)
        result = zscore_normalise(series)
        assert len(result) == 10


class TestValidateOhlcvFrame:
    def test_valid_frame_no_errors(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0], "volume": [100.0, 200.0]})
        assert validate_ohlcv_frame(df) == []

    def test_missing_close_reported(self) -> None:
        df = pd.DataFrame({"volume": [100.0, 200.0]})
        errors = validate_ohlcv_frame(df)
        assert any("close" in e for e in errors)

    def test_missing_volume_reported(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0]})
        errors = validate_ohlcv_frame(df)
        assert any("volume" in e for e in errors)

    def test_non_positive_close_reported(self) -> None:
        df = pd.DataFrame({"close": [1.0, 0.0, -1.0], "volume": [100.0, 100.0, 100.0]})
        errors = validate_ohlcv_frame(df)
        assert any("non-positive" in e for e in errors)

    def test_negative_volume_reported(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0], "volume": [100.0, -50.0]})
        errors = validate_ohlcv_frame(df)
        assert any("negative" in e for e in errors)
