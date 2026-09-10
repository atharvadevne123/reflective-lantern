"""Integration tests for the end-to-end feature → model pipeline."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from app.features import build_feature_pipeline, FEATURE_COLS
from app.model import _build_ensemble, REGIMES, compute_risk_score


def test_pipeline_output_shape(sample_ohlcv_df):
    pipeline = build_feature_pipeline()
    result = pipeline.fit_transform(sample_ohlcv_df)
    assert result.ndim == 2
    assert result.shape[1] == len(FEATURE_COLS)


def test_pipeline_no_nan_after_warmup(sample_ohlcv_df):
    pipeline = build_feature_pipeline()
    result = pipeline.fit_transform(sample_ohlcv_df)
    # Last row (after rolling windows warm up) should be finite
    assert np.all(np.isfinite(result[-1]))


def test_ensemble_predicts_valid_regime(sample_feature_array, sample_labels):
    model = _build_ensemble()
    model.fit(sample_feature_array, sample_labels)
    pred = model.predict(sample_feature_array[:1])
    assert pred[0] in range(len(REGIMES))


def test_ensemble_probabilities_sum_to_one(sample_feature_array, sample_labels):
    model = _build_ensemble()
    model.fit(sample_feature_array, sample_labels)
    proba = model.predict_proba(sample_feature_array)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("n_samples", [50, 100, 200])
def test_pipeline_scales_to_different_sizes(n_samples):
    rng = np.random.default_rng(n_samples)
    import pandas as pd
    close = 100.0 + np.cumsum(rng.normal(0, 1, n_samples))
    df = pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n_samples).astype(float),
        "market_return": rng.normal(0.0005, 0.01, n_samples),
    })
    pipeline = build_feature_pipeline()
    result = pipeline.fit_transform(df)
    assert result.shape == (n_samples, len(FEATURE_COLS))


def test_risk_score_bull_low_risk():
    score = compute_risk_score(0.1, 0.05, 0.9, "bull")
    assert score < 0.5


def test_risk_score_bear_high_risk():
    score = compute_risk_score(0.35, -0.15, 1.4, "bear")
    assert score > 0.5
