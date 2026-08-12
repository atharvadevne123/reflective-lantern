"""Tests for app/models/ensemble.py."""
from __future__ import annotations


class _FakeModel:
    _fitted = True
    def __init__(self, val):
        self._val = val
    def is_fitted(self):
        return True
    def predict(self, X):
        return [self._val] * len(X)


def test_ensemble_averages(feature_matrix):
    from app.models.ensemble import EnsemblePricingModel
    m1 = _FakeModel(10.0)
    m2 = _FakeModel(20.0)
    ens = EnsemblePricingModel([m1, m2])
    preds = ens.predict(feature_matrix)
    assert len(preds) == len(feature_matrix)
    assert all(abs(p - 15.0) < 1e-6 for p in preds)


def test_ensemble_single_model(feature_matrix):
    from app.models.ensemble import EnsemblePricingModel
    m1 = _FakeModel(42.0)
    ens = EnsemblePricingModel([m1])
    preds = ens.predict(feature_matrix)
    assert all(abs(p - 42.0) < 1e-6 for p in preds)


def test_ensemble_repr():
    from app.models.ensemble import EnsemblePricingModel
    ens = EnsemblePricingModel([_FakeModel(1.0)])
    assert "Ensemble" in repr(ens)
