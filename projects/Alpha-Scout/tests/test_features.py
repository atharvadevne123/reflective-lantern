"""Feature engineering tests."""
from __future__ import annotations

import pytest

from app.features import (
    StockFeatureEngineer,
    build_feature_pipeline,
    make_sample_ohlcv,
)


@pytest.fixture
def sample_df():
    return make_sample_ohlcv(100, ticker="TEST")


def test_make_sample_ohlcv_shape(sample_df):
    assert len(sample_df) == 100
    assert set(["close", "open", "high", "low", "volume", "ticker"]).issubset(sample_df.columns)


def test_make_sample_ohlcv_prices_positive(sample_df):
    assert (sample_df["close"] > 0).all()
    assert (sample_df["high"] >= sample_df["low"]).all()


def test_feature_engineer_produces_expected_columns(sample_df):
    eng = StockFeatureEngineer()
    eng.fit(sample_df)
    features = eng.transform(sample_df)
    for col in ["return_lag_1", "rsi_14", "macd", "volatility_5", "pe_ratio"]:
        assert col in features.columns


def test_feature_engineer_no_nan_after_fillna(sample_df):
    eng = StockFeatureEngineer()
    eng.fit(sample_df)
    features = eng.transform(sample_df)
    assert not features.isnull().any().any()


def test_feature_pipeline_returns_numpy_array(sample_df):
    pipe = build_feature_pipeline()
    pipe.fit(sample_df)
    result = pipe.transform(sample_df)
    assert hasattr(result, "shape")


@pytest.mark.parametrize("n_rows", [60, 100, 200])
def test_feature_engineer_various_lengths(n_rows):
    df = make_sample_ohlcv(n_rows, ticker="X")
    eng = StockFeatureEngineer()
    eng.fit(df)
    features = eng.transform(df)
    assert len(features) == n_rows


def test_volume_ratio_computed(sample_df):
    eng = StockFeatureEngineer()
    eng.fit(sample_df)
    features = eng.transform(sample_df)
    assert "volume_ratio" in features.columns
    assert 0.5 < features["volume_ratio"].median() < 2.0


def test_rsi_bounded(sample_df):
    eng = StockFeatureEngineer()
    eng.fit(sample_df)
    features = eng.transform(sample_df)
    rsi = features["rsi_14"].dropna()
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_macd_histogram_sign(sample_df):
    eng = StockFeatureEngineer()
    eng.fit(sample_df)
    features = eng.transform(sample_df)
    computed = (features["macd"] - features["macd_signal"]).round(6)
    assert (computed - features["macd_hist"]).abs().max() < 0.01
