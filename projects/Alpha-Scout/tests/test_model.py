"""Model training and prediction tests."""
from __future__ import annotations

import pandas as pd
import pytest

from app.features import FEATURE_COLUMNS, StockFeatureEngineer, make_sample_ohlcv
from app.model import predict_alpha, prepare_training_data, train_model


def test_prepare_training_data_returns_dataframe():
    X, y = prepare_training_data(n_tickers=1)
    assert isinstance(X, pd.DataFrame) and len(X) > 0


def test_train_model_returns_pipeline_and_metrics():
    X, y = prepare_training_data(n_tickers=2)
    _, metrics = train_model(X, y)
    assert 0.0 <= metrics["auc_mean"] <= 1.0


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT"])
def test_predict_alpha_range(trained_pipeline, ticker):
    df = make_sample_ohlcv(100, ticker=ticker)
    eng = StockFeatureEngineer()
    eng.fit(df)
    result = predict_alpha(trained_pipeline, eng.transform(df))
    assert 0 <= result["alpha_score"] <= 100


def test_model_metrics_contain_version():
    X, y = prepare_training_data(n_tickers=1)
    _, metrics = train_model(X, y)
    assert metrics["version"] == "1.0.0"
