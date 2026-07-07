"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest
from app.features import (
    AmenityCompositeTransformer,
    InvestmentPotentialTransformer,
    PropertyAgeTransformer,
    RatioFeatureTransformer,
    TierEncoderTransformer,
    build_feature_pipeline,
    extract_feature_array,
)


@pytest.fixture
def single_row():
    return pd.DataFrame([{
        "sqft": 2000.0, "bedrooms": 3, "bathrooms": 2.0, "lot_size": 6000.0,
        "year_built": 1990, "renovation_year": 2015, "condition_score": 8.0,
        "school_score": 7.0, "transit_score": 6.0, "walkability_score": 7.0,
        "crime_rate": 0.25, "median_neighborhood_price": 500_000.0,
        "median_price_per_sqft": 300.0, "avg_rental_yield": 0.07,
        "listing_days": 20, "list_price": 450_000.0,
    }])


def test_property_age_transformer(single_row):
    t = PropertyAgeTransformer(reference_year=2026)
    result = t.fit_transform(single_row)
    assert "property_age" in result.columns
    assert result["property_age"].iloc[0] == 36
    assert result["renovation_age"].iloc[0] == 11


def test_property_age_no_renovation(single_row):
    row = single_row.copy()
    row["renovation_year"] = None
    result = PropertyAgeTransformer(reference_year=2026).transform(row)
    assert result["renovation_age"].iloc[0] == result["property_age"].iloc[0]


def test_ratio_transformer(single_row):
    t = RatioFeatureTransformer()
    result = t.fit_transform(single_row)
    assert "beds_per_bath" in result.columns
    assert "sqft_per_bed" in result.columns
    assert result["sqft_per_bed"].iloc[0] == pytest.approx(2000 / 3, rel=1e-3)


def test_ratio_transformer_zero_beds(single_row):
    row = single_row.copy()
    row["bedrooms"] = 1
    result = RatioFeatureTransformer().transform(row)
    assert result["sqft_per_bed"].iloc[0] > 0


def test_amenity_composite_transformer(single_row):
    result = AmenityCompositeTransformer().fit_transform(single_row)
    assert "amenity_composite" in result.columns
    assert "risk_score" in result.columns
    assert 0 <= result["amenity_composite"].iloc[0] <= 1
    assert 0 <= result["risk_score"].iloc[0] <= 1


def test_investment_potential_transformer(single_row):
    pre = AmenityCompositeTransformer().fit_transform(single_row)
    result = InvestmentPotentialTransformer().fit_transform(pre)
    assert "investment_potential" in result.columns
    assert 0 <= result["investment_potential"].iloc[0] <= 10


def test_tier_encoder_transformer(single_row):
    pre = PropertyAgeTransformer().fit_transform(single_row)
    result = TierEncoderTransformer().fit_transform(pre)
    assert "size_tier" in result.columns
    assert "age_tier" in result.columns
    assert result["size_tier"].iloc[0] in [1, 2, 3, 4, 5]


def test_build_feature_pipeline(single_row):
    pipeline = build_feature_pipeline()
    result = pipeline.fit_transform(single_row)
    assert isinstance(result, pd.DataFrame)
    assert "amenity_composite" in result.columns


def test_extract_feature_array(single_row):
    pipeline = build_feature_pipeline()
    arr = extract_feature_array(single_row, pipeline)
    assert arr.ndim == 2
    assert arr.shape[0] == 1
    assert arr.shape[1] > 0
    assert np.all(np.isfinite(arr))


@pytest.mark.parametrize("sqft,expected_tier", [
    (500, 1), (1000, 2), (2000, 3), (3000, 4), (5000, 5),
])
def test_size_tier_values(single_row, sqft, expected_tier):
    row = single_row.copy()
    row["sqft"] = sqft
    pre = PropertyAgeTransformer().fit_transform(row)
    result = TierEncoderTransformer().fit_transform(pre)
    assert result["size_tier"].iloc[0] == expected_tier


def test_pipeline_handles_missing_values(sample_df):
    df = sample_df.copy()
    df.loc[0, "renovation_year"] = None
    pipeline = build_feature_pipeline()
    result = pipeline.fit_transform(df)
    assert not result.isnull().all().any()
