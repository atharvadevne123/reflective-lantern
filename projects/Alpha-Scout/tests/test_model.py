"""Model training and prediction tests."""
from __future__ import annotations

import pandas as pd
import pytest

from app.features import FEATURE_COLUMNS, StockFeatureEngineer, make_sample_ohlcv
from app.model import (
    predict_alpha,
    prepare_training_data,
    train_model,
)


def test_prepare_training_data_returns_dataframe():
    X, y = prepare_training_data(n_tickers=1)
    assert isinstance(X, pd.DataFrame)
    assert len(X) > 0
    assert len(X) == len(y)


def test_training_data_has_expected_features():
    X, y = prepare_training_data(n_tickers=1)
    for col in ["return_lag_1", "rsi_14", "macd", "volatility_5"]:
        assert col in X.columns, f"Missing feature: {col}"


def test_train_model_returns_pipeline_and_metrics():
    X, y = prepare_training_data(n_tickers=2)
    pipe, metrics = train_model(X, y)
    assert "auc_mean" in metrics
    assert 0.0 <= metrics["auc_mean"] <= 1.0
    assert metrics["n_features"] == X.shape[1]


def test_train_model_auc_above_random():
    X, y = prepare_training_data(n_tickers=2)
    _, metrics = train_model(X, y)
    assert metrics["auc_mean"] >= 0.4, f"AUC too low: {metrics['auc_mean']}"


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT"])
def test_predict_alpha_range(trained_pipeline, ticker):
    df = make_sample_ohlcv(100, ticker=ticker)
    eng = StockFeatureEngineer()
    eng.fit(df)
    features = eng.transform(df)
    result = predict_alpha(trained_pipeline, features)
    assert 0 <= result["alpha_score"] <= 100
    assert -0.5 <= result["return_prediction"] <= 0.5
    assert 0 <= result["confidence"] <= 1


def test_predict_alpha_has_required_keys(trained_pipeline, sample_features):
    result = predict_alpha(trained_pipeline, sample_features)
    for key in ["alpha_score", "return_prediction", "confidence", "model_version"]:
        assert key in result


def test_pipeline_predict_proba(trained_pipeline, sample_features):
    cols = [c for c in FEATURE_COLUMNS if c in sample_features.columns]
    row = sample_features[cols].fillna(0).iloc[[-1]]
    proba = trained_pipeline.predict_proba(row)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6


@pytest.mark.parametrize("n_tickers", [1, 2, 3])
def test_training_data_scales_with_tickers(n_tickers):
    X, y = prepare_training_data(n_tickers=n_tickers)
    assert len(X) > 0
    assert len(X) == len(y)
    X2, _ = prepare_training_data(n_tickers=max(1, n_tickers - 1))
    if n_tickers > 1:
        assert len(X) > len(X2)


def test_model_metrics_contain_version():
    X, y = prepare_training_data(n_tickers=1)
    _, metrics = train_model(X, y)
    assert "version" in metrics
    assert metrics["version"] == "1.0.0"
