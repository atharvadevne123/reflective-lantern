"""Feature engineering tests."""
from __future__ import annotations

import pytest

from app.features import StockFeatureEngineer, build_feature_pipeline, make_sample_ohlcv


@pytest.fixture
def sample_df():
    return make_sample_ohlcv(100, ticker="TEST")


def test_make_sample_ohlcv_shape(sample_df):
    assert len(sample_df) == 100


def test_feature_engineer_produces_expected_columns(sample_df):
    eng = StockFeatureEngineer()
    eng.fit(sample_df)
    features = eng.transform(sample_df)
    for col in ["return_lag_1", "rsi_14", "macd", "volatility_5"]:
        assert col in features.columns


def test_rsi_bounded(sample_df):
    eng = StockFeatureEngineer()
    eng.fit(sample_df)
    rsi = eng.transform(sample_df)["rsi_14"]
    assert (rsi >= 0).all() and (rsi <= 100).all()
