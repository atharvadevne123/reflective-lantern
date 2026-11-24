"""Extended feature engineering tests for signal-forge with parametrize and boundary coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    MarketCorrelationFeature,
    MomentumFeature,
    VolatilityFeature,
    VolumeRatioFeature,
)


def _price_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    volume = rng.integers(1_000, 1_000_000, n).astype(float)
    market_return = rng.normal(0, 0.01, n)
    return pd.DataFrame({"close": close, "volume": volume, "market_return": market_return})


@pytest.mark.parametrize("window", [5, 10, 21, 63])
def test_volatility_feature_window(window: int) -> None:
    df = _price_df(100)
    feat = VolatilityFeature(window=window)
    out = feat.fit_transform(df)
    assert "volatility" in out.columns
    assert out.shape[0] == 100


def test_volatility_feature_first_rows_nan() -> None:
    df = _price_df(50)
    out = VolatilityFeature(window=10).fit_transform(df)
    assert out["volatility"].iloc[:10].isna().sum() >= 1


def test_volatility_feature_non_negative_after_window() -> None:
    df = _price_df(100)
    out = VolatilityFeature(window=10).fit_transform(df)
    valid = out["volatility"].dropna()
    assert (valid >= 0).all()


def test_volatility_constant_price_is_zero() -> None:
    df = pd.DataFrame({
        "close": [100.0] * 30,
        "volume": [1_000.0] * 30,
        "market_return": [0.0] * 30,
    })
    out = VolatilityFeature(window=5).fit_transform(df)
    valid = out["volatility"].dropna()
    assert (valid == pytest.approx(0.0, abs=1e-9)).all()


@pytest.mark.parametrize("window", [5, 10, 21])
def test_momentum_feature_window(window: int) -> None:
    df = _price_df(80)
    out = MomentumFeature(window=window).fit_transform(df)
    assert "momentum" in out.columns
    assert out.shape[0] == 80


def test_momentum_feature_first_rows_nan() -> None:
    df = _price_df(50)
    out = MomentumFeature(window=10).fit_transform(df)
    assert out["momentum"].iloc[:10].isna().sum() >= 1


def test_momentum_rising_prices_positive() -> None:
    prices = np.linspace(100, 200, 50)
    df = pd.DataFrame({"close": prices, "volume": [1000.0] * 50, "market_return": [0.0] * 50})
    out = MomentumFeature(window=10).fit_transform(df)
    valid = out["momentum"].dropna()
    assert (valid > 0).all()


@pytest.mark.parametrize("window", [5, 10, 21])
def test_volume_ratio_feature_window(window: int) -> None:
    df = _price_df(80)
    out = VolumeRatioFeature(window=window).fit_transform(df)
    assert "volume_ratio" in out.columns


def test_volume_ratio_constant_volume_is_one() -> None:
    df = pd.DataFrame({
        "close": [100.0] * 30,
        "volume": [5000.0] * 30,
        "market_return": [0.0] * 30,
    })
    out = VolumeRatioFeature(window=5).fit_transform(df)
    valid = out["volume_ratio"].dropna()
    assert (valid == pytest.approx(1.0, rel=1e-4)).all()


def test_volume_ratio_no_division_by_zero() -> None:
    df = pd.DataFrame({
        "close": [100.0] * 30,
        "volume": [0.0] * 30,
        "market_return": [0.0] * 30,
    })
    out = VolumeRatioFeature(window=5).fit_transform(df)
    assert out["volume_ratio"].dropna().apply(np.isfinite).all()


@pytest.mark.parametrize("window", [5, 10, 21])
def test_market_correlation_feature_window(window: int) -> None:
    df = _price_df(80)
    out = MarketCorrelationFeature(window=window).fit_transform(df)
    assert "correlation" in out.columns


def test_market_correlation_no_market_return_col() -> None:
    df = _price_df(50).drop(columns=["market_return"])
    out = MarketCorrelationFeature(window=10).fit_transform(df)
    assert "correlation" in out.columns
    assert (out["correlation"] == 0.0).all()


def test_market_correlation_range() -> None:
    df = _price_df(100)
    out = MarketCorrelationFeature(window=10).fit_transform(df)
    valid = out["correlation"].dropna()
    assert (valid >= -1.0).all()
    assert (valid <= 1.0).all()


@pytest.mark.parametrize("n", [25, 50, 100])
def test_all_features_preserve_row_count(n: int) -> None:
    df = _price_df(n)
    for Feat in [VolatilityFeature, MomentumFeature, VolumeRatioFeature, MarketCorrelationFeature]:
        out = Feat().fit_transform(df)
        assert out.shape[0] == n
