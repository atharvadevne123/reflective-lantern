"""Model training and prediction tests for Signal-Forge."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.model import (
    _build_ensemble,
    compute_risk_score,
    find_similar_periods,
    train_model,
    REGIMES,
)


def test_ensemble_builds():
    model = _build_ensemble()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")


def test_train_model_returns_metrics(sample_feature_array, sample_labels):
    model, metrics = train_model(sample_feature_array, sample_labels, n_splits=2)
    assert "cv_auc_mean" in metrics
    assert 0.0 <= metrics["cv_auc_mean"] <= 1.0


def test_train_model_predict_probabilities(sample_feature_array, sample_labels):
    model, _ = train_model(sample_feature_array, sample_labels, n_splits=2)
    proba = model.predict_proba(sample_feature_array[:5])
    assert proba.shape == (5, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_regimes_are_valid():
    assert set(REGIMES) == {"bull", "bear", "sideways", "volatile"}


@pytest.mark.parametrize("regime,vol,mom,beta,expected_high", [
    ("bear", 0.4, -0.2, 1.5, True),
    ("bull", 0.05, 0.1, 0.8, False),
    ("volatile", 0.45, 0.25, 1.8, True),
])
def test_risk_score_regime_effect(regime, vol, mom, beta, expected_high):
    score = compute_risk_score(vol, mom, beta, regime)
    assert 0.0 <= score <= 1.0
    if expected_high:
        assert score > 0.4


def test_risk_score_clipped():
    score = compute_risk_score(10.0, 5.0, 10.0, "volatile")
    assert 0.0 <= score <= 1.0


def test_find_similar_periods_empty_index():
    result = find_similar_periods(np.ones(5, dtype=np.float32), None, [])
    assert result == []


def test_find_similar_periods_with_index():
    try:
        import faiss
        index = faiss.IndexFlatL2(5)
        vecs = np.random.randn(10, 5).astype(np.float32)
        index.add(vecs)
        meta = [{"id": i, "ticker": "X", "regime": "bull"} for i in range(10)]
        result = find_similar_periods(vecs[0], index, meta, k=3)
        assert len(result) <= 3
        assert "distance" in result[0]
    except ImportError:
        pytest.skip("faiss not available")


def test_model_predict_single_row(sample_feature_array, sample_labels):
    model, _ = train_model(sample_feature_array, sample_labels, n_splits=2)
    pred = model.predict(sample_feature_array[:1])
    assert pred[0] in range(4)
