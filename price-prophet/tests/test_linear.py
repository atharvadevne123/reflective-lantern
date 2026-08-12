"""Tests for app/models/linear.py."""
from __future__ import annotations

import pytest


def test_linear_model_not_fitted():
    from app.models.linear import LinearPricingModel
    m = LinearPricingModel()
    assert m.is_fitted() is False


def test_linear_model_fit_predict(feature_matrix):
    from app.models.linear import LinearPricingModel
    m = LinearPricingModel()
    y = [100.0, 200.0, 150.0]
    m.fit(feature_matrix, y)
    assert m.is_fitted() is True
    preds = m.predict(feature_matrix)
    assert len(preds) == len(feature_matrix)
    assert all(isinstance(p, float) for p in preds)


def test_linear_model_predict_not_fitted(feature_matrix):
    from app.models.linear import LinearPricingModel
    m = LinearPricingModel()
    with pytest.raises(Exception):
        m.predict(feature_matrix)


def test_linear_model_repr():
    from app.models.linear import LinearPricingModel
    m = LinearPricingModel()
    assert "LinearPricingModel" in repr(m)
