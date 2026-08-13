"""Tests for app/models/gradient_boost.py."""

from __future__ import annotations


def test_gradient_boost_not_fitted():
    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel()
    assert m.is_fitted() is False


def test_gradient_boost_fit_predict(feature_matrix):
    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel(n_estimators=5)
    y = [100.0, 200.0, 150.0]
    m.fit(feature_matrix, y)
    assert m.is_fitted() is True
    preds = m.predict(feature_matrix)
    assert len(preds) == len(feature_matrix)


def test_gradient_boost_repr():
    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel()
    assert "GradientBoost" in repr(m)
