"""Feature engineering tests."""

import numpy as np
import pandas as pd
import pytest

from app.features import (
    PropertyFeatureEngineer,
    build_feature_pipeline,
    generate_synthetic_data,
    get_feature_names,
    property_to_dataframe,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"bedrooms": 3, "bathrooms": 2.0, "sqft": 1500.0, "lot_size": 5000.0,
         "year_built": 2000, "neighborhood": "suburb", "property_type": "house"},
        {"bedrooms": 1, "bathrooms": 1.0, "sqft": 500.0, "lot_size": 1000.0,
         "year_built": 1980, "neighborhood": "downtown", "property_type": "studio"},
    ])


def test_feature_engineer_fit_transform(sample_df):
    eng = PropertyFeatureEngineer()
    result = eng.fit(sample_df).transform(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_feature_names_match_output(sample_df):
    eng = PropertyFeatureEngineer()
    result = eng.fit(sample_df).transform(sample_df)
    expected = get_feature_names()
    assert list(result.columns) == expected


def test_property_age_computed(sample_df):
    eng = PropertyFeatureEngineer()
    result = eng.fit(sample_df).transform(sample_df)
    # property_age col is index 3 after scaler — check via raw transform
    raw = eng.transform(sample_df)
    assert "property_age" in raw.columns


def test_unknown_neighborhood_falls_back(sample_df):
    eng = PropertyFeatureEngineer()
    eng.fit(sample_df)
    unseen = sample_df.copy()
    unseen["neighborhood"] = "atlantis"
    result = eng.transform(unseen)
    assert result.shape[0] == 2


def test_build_feature_pipeline_runs(sample_df):
    pipe = build_feature_pipeline()
    pipe.fit(sample_df)
    result = pipe.transform(sample_df)
    assert result.shape[0] == 2
    assert result.shape[1] == len(get_feature_names())


def test_generate_synthetic_data_shape():
    X, y_price, y_rental = generate_synthetic_data(n=100)
    assert len(X) == 100
    assert len(y_price) == 100
    assert len(y_rental) == 100


def test_prices_positive():
    _, y_price, _ = generate_synthetic_data(n=200)
    assert (y_price > 0).all()


def test_rental_yields_bounded():
    _, _, y_rental = generate_synthetic_data(n=200)
    assert (y_rental >= 0.01).all()
    assert (y_rental <= 0.20).all()


def test_property_to_dataframe():
    data = {
        "bedrooms": 2, "bathrooms": 1.5, "sqft": 900.0, "lot_size": 3000.0,
        "year_built": 1995, "neighborhood": "midtown", "property_type": "condo",
    }
    df = property_to_dataframe(data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


@pytest.mark.parametrize("beds,sqft", [(1, 400), (3, 1500), (5, 3200)])
def test_sqft_per_bedroom_computed(beds, sqft):
    df = pd.DataFrame([{
        "bedrooms": beds, "bathrooms": 1.0, "sqft": float(sqft), "lot_size": 5000.0,
        "year_built": 2000, "neighborhood": "suburb", "property_type": "house",
    }])
    eng = PropertyFeatureEngineer()
    eng.fit(df)
    result = eng.transform(df)
    assert "sqft_per_bedroom" in result.columns
    expected = sqft / beds
    assert abs(result["sqft_per_bedroom"].iloc[0] - expected) < 1e-6
