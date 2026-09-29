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


def test_synthetic_data_delivery_minutes_positive():
    df = generate_synthetic_data(n=200, seed=1)
    assert (df["delivery_minutes"] > 0).all()


def test_prepare_X_fit_false_after_fit(sample_df):
    """transform-only pass must not crash after the pipeline is fitted."""
    pipe = build_feature_pipeline()
    prepare_X(sample_df, pipe, fit=True)
    X2 = prepare_X(sample_df.head(5), pipe, fit=False)
    assert X2.shape[1] == len(FEATURE_COLS)


def test_unknown_carrier_falls_back(sample_df):
    """Unknown carrier should not raise during transform."""
    pipe = build_feature_pipeline()
    prepare_X(sample_df, pipe, fit=True)
    row = sample_df.head(1).copy()
    row["carrier"] = "UNKNOWN_XYZ"
    X = prepare_X(row, pipe, fit=False)
    assert X.shape[1] == len(FEATURE_COLS)


def test_is_peak_hour_flag():
    from app.features import TemporalFeatureExtractor

    df = pd.DataFrame({"hour_of_day": [8, 12, 17, 20], "day_of_week": [1, 1, 1, 1]})
    tfe = TemporalFeatureExtractor()
    out = tfe.fit_transform(df)
    assert out.loc[0, "is_peak"] == 1  # 8 AM peak
    assert out.loc[1, "is_peak"] == 0  # 12 PM non-peak
    assert out.loc[2, "is_peak"] == 1  # 5 PM peak
    assert out.loc[3, "is_peak"] == 0  # 8 PM non-peak


def test_cyclical_encoding_symmetry():
    """hour_sin for hour 0 and hour 24 (mod 24 = 0) should be the same."""
    from app.features import TemporalFeatureExtractor

    df = pd.DataFrame({"hour_of_day": [0, 12], "day_of_week": [0, 0]})
    tfe = TemporalFeatureExtractor()
    out = tfe.fit_transform(df)
    assert abs(out.loc[0, "hour_cos"] - 1.0) < 1e-9
    assert abs(out.loc[1, "hour_cos"] - (-1.0)) < 1e-9
