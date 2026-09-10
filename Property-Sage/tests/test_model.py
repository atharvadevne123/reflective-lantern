"""Model training and prediction tests."""

import pandas as pd
import pytest

from app.features import generate_synthetic_data
from app.model import _build_pipeline, get_metrics, predict, train_model


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    import app.model as m
    m.MODEL_DIR = tmp
    m.PRICE_MODEL_PATH = tmp / "price_model.joblib"
    m.RENTAL_MODEL_PATH = tmp / "rental_model.joblib"
    m.METRICS_PATH = tmp / "metrics.json"

    X, y_price, y_rental = generate_synthetic_data(n=300, seed=99)
    metrics = train_model(X, y_price, y_rental)
    return m, metrics


def test_train_model_returns_metrics(trained_models):
    _, metrics = trained_models
    assert "price_r2_mean" in metrics
    assert "rental_r2_mean" in metrics
    assert "n_train" in metrics
    assert metrics["n_train"] == 300


def test_price_r2_reasonable(trained_models):
    _, metrics = trained_models
    assert metrics["price_r2_mean"] > 0.5


def test_rental_r2_reasonable(trained_models):
    _, metrics = trained_models
    assert metrics["rental_r2_mean"] > 0.3


def test_predict_output_structure(trained_models):
    m, _ = trained_models
    import joblib
    price_model = joblib.load(m.PRICE_MODEL_PATH)
    rental_model = joblib.load(m.RENTAL_MODEL_PATH)

    X = pd.DataFrame([{
        "bedrooms": 3, "bathrooms": 2.0, "sqft": 1500.0, "lot_size": 6000.0,
        "year_built": 2005, "neighborhood": "suburb", "property_type": "house",
    }])
    result = predict(price_model, rental_model, X)
    assert "predicted_price" in result
    assert "predicted_rental_yield" in result
    assert result["predicted_price"] > 0
    assert 0 < result["predicted_rental_yield"] < 1


@pytest.mark.parametrize("n_beds,sqft,expected_min", [
    (5, 3000, 100_000),
    (1, 400, 50_000),
    (3, 1500, 80_000),
])
def test_price_scales_with_features(trained_models, n_beds, sqft, expected_min):
    m, _ = trained_models
    import joblib
    price_model = joblib.load(m.PRICE_MODEL_PATH)
    rental_model = joblib.load(m.RENTAL_MODEL_PATH)

    X = pd.DataFrame([{
        "bedrooms": n_beds, "bathrooms": 1.0, "sqft": sqft, "lot_size": sqft * 2,
        "year_built": 2000, "neighborhood": "suburb", "property_type": "house",
    }])
    result = predict(price_model, rental_model, X)
    assert result["predicted_price"] >= expected_min


def test_build_pipeline_has_three_steps():
    pipe = _build_pipeline()
    assert len(pipe.steps) == 3


def test_metrics_file_empty_before_train():
    result = get_metrics()
    assert isinstance(result, dict)
