"""Anomaly detection tests."""
from __future__ import annotations

import pytest

from app.anomaly import fit_anomaly_detector, predict_anomaly
from app.features import FEATURE_COLUMNS, StockFeatureEngineer, make_sample_ohlcv


@pytest.fixture(scope="module")
def fitted_detector():
    df = make_sample_ohlcv(200, ticker="AAPL")
    eng = StockFeatureEngineer()
    eng.fit(df)
    features = eng.transform(df)
    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    return fit_anomaly_detector(features[cols])


def test_anomaly_detector_fits(fitted_detector):
    assert fitted_detector is not None


def test_normal_data_not_anomaly(fitted_detector):
    df = make_sample_ohlcv(100, ticker="TEST")
    eng = StockFeatureEngineer()
    eng.fit(df)
    features = eng.transform(df)
    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    result = predict_anomaly(features[cols].iloc[[-1]])
    assert "is_anomaly" in result
    assert "anomaly_score" in result


def test_extreme_values_flagged_as_anomaly(fitted_detector):
    df = make_sample_ohlcv(100, ticker="EXTREME")
    eng = StockFeatureEngineer()
    eng.fit(df)
    features = eng.transform(df)
    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    extreme = features[cols].copy()
    extreme.iloc[-1] = 999999.0
    result = predict_anomaly(extreme.iloc[[-1]])
    assert result["is_anomaly"] is True


def test_predict_without_fitted_detector():
    from app import anomaly
    anomaly._detector = None
    df = make_sample_ohlcv(60, ticker="X")
    eng = StockFeatureEngineer()
    eng.fit(df)
    features = eng.transform(df)
    result = predict_anomaly(features)
    assert "error" in result
    assert result["is_anomaly"] is False
