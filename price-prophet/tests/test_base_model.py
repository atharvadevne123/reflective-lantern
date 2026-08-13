"""Tests for app/models/base.py."""
from __future__ import annotations


class ConcreteModel:
    """Minimal concrete subclass for testing BasePricingModel."""
    _fitted: bool = False

    def fit(self, X, y):
        self._fitted = True

    def predict(self, X):
        return [1.0] * len(X)

    def is_fitted(self):
        return bool(self._fitted)

    def __repr__(self):
        return f"ConcreteModel({'fitted' if self._fitted else 'not fitted'})"


def test_is_fitted_initially_false():
    m = ConcreteModel()
    assert m.is_fitted() is False


def test_is_fitted_after_fit():
    m = ConcreteModel()
    m.fit([[1.0]], [2.0])
    assert m.is_fitted() is True


def test_predict_returns_list():
    m = ConcreteModel()
    result = m.predict([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(result, list)
    assert len(result) == 2


def test_repr_not_fitted():
    m = ConcreteModel()
    assert "not fitted" in repr(m)


def test_repr_fitted():
    m = ConcreteModel()
    m.fit([], [])
    assert "fitted" in repr(m)
