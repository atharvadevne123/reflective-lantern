"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    FEATURE_NAMES,
    LagRollingFeatures,
    RatioFeatures,
    TechnicalIndicators,
    engineer_features,
)


@pytest.fixture()
def ohlcv_df():
    rng = np.random.default_rng(42)
    n = 100
    close = 150 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, n),
            "high": close + rng.uniform(0.5, 2, n),
            "low": close - rng.uniform(0.5, 2, n),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


def test_technical_indicators_shape(ohlcv_df):
    result = TechnicalIndicators().transform(ohlcv_df)
    assert result.shape == (len(ohlcv_df), 8)


def test_technical_indicators_no_nan(ohlcv_df):
    result = TechnicalIndicators().transform(ohlcv_df)
    assert not np.any(np.isnan(result))


@pytest.mark.parametrize("rsi_period", [7, 14, 21])
def test_rsi_bounds(ohlcv_df, rsi_period):
    ti = TechnicalIndicators(rsi_period=rsi_period)
    result = ti.transform(ohlcv_df)
    rsi_col = result[:, 0]
    assert np.all(rsi_col >= 0)
    assert np.all(rsi_col <= 100)


def test_lag_rolling_shape(ohlcv_df):
    result = LagRollingFeatures().transform(ohlcv_df)
    assert result.shape == (len(ohlcv_df), 11)


def test_lag_rolling_no_nan(ohlcv_df):
    result = LagRollingFeatures().transform(ohlcv_df)
    assert not np.any(np.isnan(result))


def test_ratio_features_shape(ohlcv_df):
    result = RatioFeatures().transform(ohlcv_df)
    assert result.shape == (len(ohlcv_df), 3)


def test_engineer_features_total_shape(ohlcv_df):
    features = engineer_features(ohlcv_df)
    assert features.shape == (len(ohlcv_df), len(FEATURE_NAMES))


def test_engineer_features_no_nan(ohlcv_df):
    features = engineer_features(ohlcv_df)
    assert not np.any(np.isnan(features))


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 22


def test_price_position_in_zero_one(ohlcv_df):
    result = RatioFeatures().transform(ohlcv_df)
    price_pos = result[:, 2]
    assert np.all(price_pos >= 0)
    assert np.all(price_pos <= 1)


@pytest.mark.parametrize("n_rows", [60, 80, 100])
def test_engineer_features_different_lengths(n_rows):
    rng = np.random.default_rng(n_rows)
    close = 100 + np.cumsum(rng.normal(0, 1, n_rows))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(n_rows) * 1e6,
        }
    )
    result = engineer_features(df)
    assert result.shape == (n_rows, 22)
