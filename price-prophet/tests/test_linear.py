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


def test_linear_model_coef_empty_before_fit() -> None:
    from app.models.linear import LinearPricingModel

    m = LinearPricingModel()
    assert m.coef_ == []


def test_linear_model_predict_raises_before_fit() -> None:
    import pytest

    from app.models.linear import LinearPricingModel

    m = LinearPricingModel()
    with pytest.raises(RuntimeError):
        m.predict([[1.0, 2.0]])


def test_linear_model_no_intercept_fit() -> None:
    from app.models.linear import LinearPricingModel

    m = LinearPricingModel(fit_intercept=False)
    X = [[1.0], [2.0], [3.0]]
    y = [2.0, 4.0, 6.0]
    m.fit(X, y)
    preds = m.predict(X)
    assert all(abs(p - e) < 0.5 for p, e in zip(preds, y, strict=False))


def test_linear_model_repr_contains_fitted_state() -> None:
    from app.models.linear import LinearPricingModel

    m = LinearPricingModel()
    assert "not fitted" in repr(m)
    m.fit([[1.0]], [1.0])
    assert "fitted" in repr(m)


@pytest.mark.parametrize("n", [5, 20, 50])
def test_linear_model_predict_count_matches_input(n: int) -> None:
    from app.models.linear import LinearPricingModel

    m = LinearPricingModel()
    X_train = [[float(i)] for i in range(n)]
    y_train = [float(i) * 3 for i in range(n)]
    m.fit(X_train, y_train)
    preds = m.predict(X_train)
    assert len(preds) == n


def test_linear_model_predictions_are_floats() -> None:
    from app.models.linear import LinearPricingModel

    m = LinearPricingModel()
    X = [[1.0], [2.0], [3.0]]
    y = [2.0, 4.0, 6.0]
    m.fit(X, y)
    preds = m.predict(X)
    assert all(isinstance(p, float) for p in preds)
