"""Tests for feature engineering pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    CARRIERS,
    FEATURE_COLS,
    build_feature_pipeline,
    generate_synthetic_data,
    prepare_X,
)


def test_synthetic_data_has_expected_columns(sample_df):
    assert "carrier" in sample_df.columns
    assert "delivery_minutes" in sample_df.columns


def test_pipeline_adds_temporal_features(sample_df):
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(sample_df)
    assert "hour_sin" in out.columns
    assert "is_weekend" in out.columns
    assert "is_peak" in out.columns


def test_pipeline_adds_route_features(sample_df):
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(sample_df)
    assert "distance_bucket" in out.columns
    assert "weight_per_km" in out.columns
    assert "carrier_risk" in out.columns


def test_pipeline_encodes_carrier(sample_df):
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(sample_df)
    assert "carrier_enc" in out.columns
    assert out["carrier_enc"].dtype in (int, np.int64, np.int32)


def test_prepare_X_returns_correct_columns(sample_df):
    pipe = build_feature_pipeline()
    X = prepare_X(sample_df, pipe, fit=True)
    assert X.shape[1] == len(FEATURE_COLS)


def test_prepare_X_no_nans(sample_df):
    pipe = build_feature_pipeline()
    X = prepare_X(sample_df, pipe, fit=True)
    assert not np.isnan(X).any()


def test_weight_per_km_positive(sample_df):
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(sample_df)
    assert (out["weight_per_km"] > 0).all()


def test_hour_sin_bounded(sample_df):
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(sample_df)
    assert out["hour_sin"].between(-1.0, 1.0).all()


def test_is_weekend_flag(sample_df):
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(sample_df)
    # Day 5 and 6 should be weekend
    weekend_rows = out[sample_df["day_of_week"].isin([5, 6])]
    assert (weekend_rows["is_weekend"] == 1).all()


def test_distance_bucket_range(sample_df):
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(sample_df)
    assert out["distance_bucket"].between(0, 3).all()


@pytest.mark.parametrize("carrier", CARRIERS)
def test_all_carriers_have_risk_score(carrier):
    from app.features import RouteFeatureEngineer

    df = pd.DataFrame(
        {
            "carrier": [carrier],
            "distance_km": [50.0],
            "weight_kg": [5.0],
            "route_type": ["urban"],
            "hour_of_day": [12],
            "day_of_week": [1],
        }
    )
    eng = RouteFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["carrier_risk"].iloc[0] > 0


def test_generate_synthetic_data_shape():
    df = generate_synthetic_data(n=300, seed=0)
    assert df.shape[0] == 300
    assert "delivery_minutes" in df.columns
