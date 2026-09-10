"""Feature engineering pipeline tests for Signal-Forge."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.features import (
    FEATURE_COLS,
    BetaFeature,
    DropRawColumns,
    MarketCorrelationFeature,
    MomentumFeature,
    VolatilityFeature,
    VolumeRatioFeature,
    build_feature_pipeline,
    extract_single_row,
)


def test_volatility_feature(sample_ohlcv_df):
    tf = VolatilityFeature(window=5)
    result = tf.fit_transform(sample_ohlcv_df)
    assert "volatility" in result.columns
    non_null = result["volatility"].dropna()
    assert len(non_null) > 0
    assert (non_null >= 0).all()


def test_momentum_feature(sample_ohlcv_df):
    tf = MomentumFeature(window=5)
    result = tf.fit_transform(sample_ohlcv_df)
    assert "momentum" in result.columns


def test_volume_ratio_feature(sample_ohlcv_df):
    tf = VolumeRatioFeature(window=5)
    result = tf.fit_transform(sample_ohlcv_df)
    assert "volume_ratio" in result.columns
    non_null = result["volume_ratio"].dropna()
    assert (non_null >= 0).all()


def test_correlation_feature_without_market(sample_ohlcv_df):
    df = sample_ohlcv_df.drop(columns=["market_return"])
    tf = MarketCorrelationFeature()
    result = tf.fit_transform(df)
    assert "correlation" in result.columns
    assert (result["correlation"] == 0.0).all()


def test_beta_feature_without_market(sample_ohlcv_df):
    df = sample_ohlcv_df.drop(columns=["market_return"])
    tf = BetaFeature()
    result = tf.fit_transform(df)
    assert "beta" in result.columns
    assert (result["beta"] == 1.0).all()


def test_drop_raw_columns(sample_ohlcv_df):
    df = sample_ohlcv_df.copy()
    tf = DropRawColumns()
    result = tf.fit_transform(df)
    raw_cols = {"open", "high", "low", "close", "volume", "market_return"}
    assert not raw_cols.intersection(result.columns)


def test_build_pipeline_returns_array(sample_ohlcv_df):
    pipeline = build_feature_pipeline()
    result = pipeline.fit_transform(sample_ohlcv_df)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == len(FEATURE_COLS)


def test_extract_single_row_valid():
    row = {"close": 150.0, "volume": 5_000_000.0}
    df = extract_single_row(row)
    assert len(df) == 22
    assert "close" in df.columns
    assert "volume" in df.columns


def test_extract_single_row_missing_close():
    with pytest.raises(ValueError, match="close"):
        extract_single_row({"volume": 1_000_000.0})


def test_extract_single_row_missing_volume():
    with pytest.raises(ValueError, match="volume"):
        extract_single_row({"close": 100.0})


@pytest.mark.parametrize("window", [5, 10, 21])
def test_volatility_different_windows(sample_ohlcv_df, window):
    tf = VolatilityFeature(window=window)
    result = tf.fit_transform(sample_ohlcv_df)
    assert "volatility" in result.columns


@pytest.mark.parametrize("window", [5, 10, 21])
def test_momentum_different_windows(sample_ohlcv_df, window):
    tf = MomentumFeature(window=window)
    result = tf.fit_transform(sample_ohlcv_df)
    assert "momentum" in result.columns
