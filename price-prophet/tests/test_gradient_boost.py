"""Tests for app/models/gradient_boost.py."""

from __future__ import annotations

import pytest


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


def test_gradient_boost_feature_importances_empty_before_fit() -> None:
    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel()
    assert m.feature_importances_ == []


def test_gradient_boost_predict_raises_before_fit() -> None:
    import pytest

    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel()
    with pytest.raises(RuntimeError):
        m.predict([[1.0, 2.0]])


def test_gradient_boost_repr_contains_params() -> None:
    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel(n_estimators=50, max_depth=3)
    r = repr(m)
    assert "50" in r
    assert "3" in r


@pytest.mark.parametrize("n_estimators", [10, 50])
def test_gradient_boost_fit_different_estimators(n_estimators: int) -> None:
    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel(n_estimators=n_estimators)
    X = [[float(i)] for i in range(20)]
    y = [float(i) * 2 for i in range(20)]
    m.fit(X, y)
    assert m.is_fitted()


@pytest.mark.parametrize("max_depth", [2, 4, 6])
def test_gradient_boost_predict_returns_list(max_depth: int) -> None:
    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel(n_estimators=10, max_depth=max_depth)
    X = [[float(i)] for i in range(20)]
    y = [float(i) * 1.5 for i in range(20)]
    m.fit(X, y)
    preds = m.predict([[5.0], [10.0]])
    assert isinstance(preds, list)
    assert len(preds) == 2


def test_gradient_boost_not_fitted_initially() -> None:
    from app.models.gradient_boost import GradientBoostPricingModel

    m = GradientBoostPricingModel()
    assert m.is_fitted() is False
