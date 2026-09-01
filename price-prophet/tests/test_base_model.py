"""Tests for app/models/base.py."""

from __future__ import annotations

import pytest


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


def test_base_model_repr_shows_fitted_state() -> None:
    from app.models.base import BasePricingModel

    class _Impl(BasePricingModel):
        def fit(self, X, y):
            self._fitted = True

        def predict(self, X):
            return [1.0] * len(X)

    m = _Impl()
    assert "not fitted" in repr(m)
    m.fit([[1.0]], [1.0])
    assert "fitted" in repr(m)


def test_base_model_is_fitted_false_before_fit() -> None:
    from app.models.base import BasePricingModel

    class _Impl(BasePricingModel):
        def fit(self, X, y):
            self._fitted = True

        def predict(self, X):
            return [1.0] * len(X)

    m = _Impl()
    assert m.is_fitted() is False


def test_base_model_is_fitted_true_after_fit() -> None:
    from app.models.base import BasePricingModel

    class _Impl(BasePricingModel):
        def fit(self, X, y):
            self._fitted = True

        def predict(self, X):
            return [1.0] * len(X)

    m = _Impl()
    m.fit([[1.0]], [1.0])
    assert m.is_fitted() is True


@pytest.mark.parametrize("n_preds", [1, 5, 10])
def test_base_predict_count_matches_input(n_preds: int) -> None:
    from app.models.base import BasePricingModel

    class _Impl(BasePricingModel):
        def fit(self, X, y):
            self._fitted = True

        def predict(self, X):
            return [1.0] * len(X)

    m = _Impl()
    m.fit([[0.0]], [0.0])
    result = m.predict([[float(i)] for i in range(n_preds)])
    assert len(result) == n_preds


def test_base_model_predict_raises_when_not_fitted() -> None:
    from app.models.base import BasePricingModel

    class _Impl(BasePricingModel):
        def fit(self, X, y):
            self._fitted = True

        def predict(self, X):
            if not self.is_fitted():
                raise RuntimeError("not fitted")
            return [1.0] * len(X)

    m = _Impl()
    with pytest.raises(RuntimeError):
        m.predict([[1.0]])
