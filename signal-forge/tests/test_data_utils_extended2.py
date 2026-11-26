"""Extended data utility tests — second pass with parametrize and boundary coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data_utils import (
    compute_log_returns,
    fill_missing_ohlcv,
    validate_ohlcv_frame,
    winsorise,
    zscore_normalise,
)


def _make_ohlcv(n: int = 10, with_nans: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [105.0 + i for i in range(n)],
            "low": [95.0 + i for i in range(n)],
            "close": [102.0 + i for i in range(n)],
            "volume": [1000.0 * (i + 1) for i in range(n)],
        }
    )
    if with_nans:
        df.loc[2, "close"] = float("nan")
        df.loc[5, "volume"] = float("nan")
    return df


def test_fill_missing_ohlcv_no_nans_unchanged() -> None:
    df = _make_ohlcv(n=10)
    result = fill_missing_ohlcv(df)
    assert result.isnull().sum().sum() == 0


def test_fill_missing_ohlcv_fills_nans() -> None:
    df = _make_ohlcv(n=10, with_nans=True)
    result = fill_missing_ohlcv(df)
    assert result["close"].isnull().sum() == 0


def test_fill_missing_ohlcv_does_not_mutate_original() -> None:
    df = _make_ohlcv(n=10, with_nans=True)
    original_nan_count = df.isnull().sum().sum()
    fill_missing_ohlcv(df)
    assert df.isnull().sum().sum() == original_nan_count


def test_compute_log_returns_length() -> None:
    prices = pd.Series([100.0, 110.0, 105.0, 115.0])
    returns = compute_log_returns(prices)
    assert len(returns) == len(prices)


def test_compute_log_returns_first_is_nan() -> None:
    prices = pd.Series([100.0, 110.0, 120.0])
    returns = compute_log_returns(prices)
    assert pd.isna(returns.iloc[0])


def test_compute_log_returns_rising_prices_positive() -> None:
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])
    returns = compute_log_returns(prices)
    assert all(returns.iloc[1:] > 0)


def test_winsorise_clips_extremes() -> None:
    series = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    result = winsorise(series, lower=0.1, upper=0.9)
    assert result.max() < 100.0


def test_winsorise_does_not_change_middle_values() -> None:
    series = pd.Series(list(range(1, 101)), dtype=float)
    result = winsorise(series, lower=0.01, upper=0.99)
    assert result.iloc[50] == pytest.approx(series.iloc[50])


@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.01, 0.99),
        (0.05, 0.95),
        (0.10, 0.90),
    ],
)
def test_winsorise_various_quantiles(lower: float, upper: float) -> None:
    series = pd.Series(list(range(1, 101)), dtype=float)
    result = winsorise(series, lower=lower, upper=upper)
    assert len(result) == len(series)


def test_zscore_normalise_mean_near_zero() -> None:
    series = pd.Series([float(i) for i in range(100)])
    result = zscore_normalise(series)
    assert abs(result.mean()) < 1e-6


def test_zscore_normalise_std_near_one() -> None:
    series = pd.Series([float(i) for i in range(100)])
    result = zscore_normalise(series)
    assert abs(result.std() - 1.0) < 0.1


def test_zscore_normalise_zero_std_returns_zeros() -> None:
    series = pd.Series([5.0] * 10)
    result = zscore_normalise(series)
    assert (result == 0.0).all()


def test_validate_ohlcv_frame_valid_no_errors() -> None:
    df = _make_ohlcv(10)
    errors = validate_ohlcv_frame(df)
    assert errors == []


def test_validate_ohlcv_frame_missing_close_column() -> None:
    df = _make_ohlcv(10).drop(columns=["close"])
    errors = validate_ohlcv_frame(df)
    assert any("close" in e for e in errors)


def test_validate_ohlcv_frame_negative_close_raises_error() -> None:
    df = _make_ohlcv(5)
    df.loc[2, "close"] = -1.0
    errors = validate_ohlcv_frame(df)
    assert errors
