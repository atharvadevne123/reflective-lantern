"""Tests for batch_predict helper."""

import pandas as pd
import pytest

from app.features import generate_synthetic_data
from app.model import batch_predict, train_model


@pytest.fixture(scope="module")
def batch_models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("batch_models")
    import app.model as m
    m.MODEL_DIR = tmp
    m.PRICE_MODEL_PATH = tmp / "price_model.joblib"
    m.RENTAL_MODEL_PATH = tmp / "rental_model.joblib"
    m.METRICS_PATH = tmp / "metrics.json"
    X, y_price, y_rental = generate_synthetic_data(n=200, seed=7)
    train_model(X, y_price, y_rental)
    import joblib
    return joblib.load(m.PRICE_MODEL_PATH), joblib.load(m.RENTAL_MODEL_PATH)


def test_batch_predict_returns_list(batch_models):
    price_model, rental_model = batch_models
    X, _, _ = generate_synthetic_data(n=5, seed=42)
    results = batch_predict(price_model, rental_model, X)
    assert isinstance(results, list)
    assert len(results) == 5


def test_batch_predict_keys(batch_models):
    price_model, rental_model = batch_models
    X, _, _ = generate_synthetic_data(n=3, seed=1)
    for r in batch_predict(price_model, rental_model, X):
        assert "predicted_price" in r
        assert "predicted_rental_yield" in r
        assert "estimated_monthly_rental" in r


@pytest.mark.parametrize("n", [1, 5, 10])
def test_batch_predict_size(batch_models, n):
    price_model, rental_model = batch_models
    X, _, _ = generate_synthetic_data(n=n, seed=0)
    assert len(batch_predict(price_model, rental_model, X)) == n
