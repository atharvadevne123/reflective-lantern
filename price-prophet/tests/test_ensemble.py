"""Tests for app/models/ensemble.py."""

from __future__ import annotations

import pytest


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


def test_ensemble_member_count(feature_matrix) -> None:
    from app.models.ensemble import EnsemblePricingModel

    ens = EnsemblePricingModel([_FakeModel(1.0), _FakeModel(2.0), _FakeModel(3.0)])
    assert ens.member_count() == 3


def test_ensemble_remove_model(feature_matrix) -> None:
    from app.models.ensemble import EnsemblePricingModel

    ens = EnsemblePricingModel([_FakeModel(10.0), _FakeModel(20.0)])
    removed = ens.remove_model(0)
    assert ens.member_count() == 1
    assert removed is not None


def test_ensemble_remove_last_raises(feature_matrix) -> None:
    from app.models.ensemble import EnsemblePricingModel

    ens = EnsemblePricingModel([_FakeModel(10.0)])
    with pytest.raises(IndexError):
        ens.remove_model(0)


def test_ensemble_contains_in_repr(feature_matrix) -> None:
    from app.models.ensemble import EnsemblePricingModel

    ens = EnsemblePricingModel([_FakeModel(1.0)])
    r = repr(ens)
    assert "fitted" in r or "not fitted" in r


@pytest.mark.parametrize("n_models", [1, 2, 3])
def test_ensemble_member_count_parametrized(feature_matrix, n_models: int) -> None:
    from app.models.ensemble import EnsemblePricingModel

    models = [_FakeModel(float(i)) for i in range(n_models)]
    ens = EnsemblePricingModel(models)
    assert ens.member_count() == n_models


def test_ensemble_predict_output_count_matches_input(feature_matrix) -> None:
    from app.models.ensemble import EnsemblePricingModel

    ens = EnsemblePricingModel([_FakeModel(5.0), _FakeModel(10.0)])
    X = feature_matrix[:3]
    preds = ens.predict(X)
    assert len(preds) == 3
