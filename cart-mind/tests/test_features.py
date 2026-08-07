"""Tests for the Cart-Mind feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    INTERACTION_COLS,
    ITEM_COLS,
    USER_COLS,
    DiscountEncoder,
    InteractionFeatures,
    LagRollingFeatures,
    RatioFeatures,
    build_feature_pipeline,
    make_sample_dataframe,
)

_ALL_COLS = USER_COLS + ITEM_COLS + INTERACTION_COLS


class TestRatioFeatures:
    def test_adds_new_columns(self, sample_df):
        tf = RatioFeatures()
        out = tf.fit_transform(sample_df)
        for col in ("price_per_rating", "value_score", "engagement_rate"):
            assert col in out.columns

    def test_no_infs_in_output(self, sample_df):
        out = RatioFeatures().fit_transform(sample_df)
        assert not out.isin([np.inf, -np.inf]).any().any()

    def test_fit_returns_self(self, sample_df):
        tf = RatioFeatures()
        result = tf.fit(sample_df)
        assert result is tf


class TestInteractionFeatures:
    def test_adds_affinity_score(self, sample_df):
        out = InteractionFeatures().fit_transform(sample_df)
        assert "affinity_score" in out.columns

    def test_recency_weight_between_0_and_1(self, sample_df):
        out = InteractionFeatures().fit_transform(sample_df)
        assert out["recency_weight"].between(0, 1).all()

    def test_price_sensitivity_positive(self, sample_df):
        out = InteractionFeatures().fit_transform(sample_df)
        assert (out["price_sensitivity"] >= 0).all()


class TestLagRollingFeatures:
    def test_stores_mean_during_fit(self, sample_df):
        tf = LagRollingFeatures()
        tf.fit(sample_df)
        assert hasattr(tf, "session_mean_")
        assert hasattr(tf, "order_mean_")

    def test_session_norm_non_negative(self, sample_df):
        out = LagRollingFeatures().fit_transform(sample_df)
        assert (out["session_norm"] >= 0).all()

    def test_purchase_velocity_non_negative(self, sample_df):
        out = LagRollingFeatures().fit_transform(sample_df)
        assert (out["purchase_velocity"] >= 0).all()


class TestDiscountEncoder:
    def test_adds_discount_bucket(self, sample_df):
        out = DiscountEncoder().fit_transform(sample_df)
        assert "discount_bucket" in out.columns

    def test_inventory_pressure_binary(self, sample_df):
        out = DiscountEncoder().fit_transform(sample_df)
        assert out["inventory_pressure"].isin([0.0, 1.0]).all()

    @pytest.mark.parametrize("pct,expected_bucket", [(0, 0), (10, 1), (20, 2), (40, 3), (60, 4)])
    def test_discount_bucket_values(self, sample_df, pct, expected_bucket):
        df = sample_df.copy()
        df["item_discount_pct"] = float(pct)
        out = DiscountEncoder().fit_transform(df)
        assert int(out["discount_bucket"].iloc[0]) == expected_bucket


class TestBuildFeaturePipeline:
    def test_pipeline_transforms_without_error(self, sample_df):
        X = sample_df[_ALL_COLS]
        pipe = build_feature_pipeline()
        out = pipe.fit_transform(X)
        assert out is not None

    def test_output_has_no_nans(self, sample_df):
        X = sample_df[_ALL_COLS]
        pipe = build_feature_pipeline()
        out = pipe.fit_transform(X)
        assert not np.isnan(out).any()

    def test_pipeline_has_five_stages(self):
        pipe = build_feature_pipeline()
        assert len(pipe.steps) == 5

    def test_pipeline_output_is_numpy(self, sample_df):
        X = sample_df[_ALL_COLS]
        pipe = build_feature_pipeline()
        out = pipe.fit_transform(X)
        assert isinstance(out, np.ndarray)


class TestMakeSampleDataframe:
    def test_returns_correct_shape(self):
        df = make_sample_dataframe(n=50)
        assert len(df) == 50

    def test_columns_present(self):
        df = make_sample_dataframe(n=30)
        for col in _ALL_COLS:
            assert col in df.columns

    def test_reproducible_with_same_seed(self):
        df1 = make_sample_dataframe(n=20, seed=7)
        df2 = make_sample_dataframe(n=20, seed=7)
        pd.testing.assert_frame_equal(df1, df2)


class TestMakePurchaseLabels:
    def test_returns_binary_series(self, sample_df):
        from app.features import make_purchase_labels

        y = make_purchase_labels(sample_df)
        assert set(y.unique()) <= {0, 1}

    def test_length_matches_input(self, sample_df):
        from app.features import make_purchase_labels

        y = make_purchase_labels(sample_df)
        assert len(y) == len(sample_df)

    def test_both_classes_present(self):
        from app.features import make_purchase_labels, make_sample_dataframe

        df = make_sample_dataframe(n=500, seed=3)
        y = make_purchase_labels(df, seed=3)
        assert y.nunique() == 2

    def test_reproducible_with_same_seed(self, sample_df):
        from app.features import make_purchase_labels

        y1 = make_purchase_labels(sample_df, seed=11)
        y2 = make_purchase_labels(sample_df, seed=11)
        pd.testing.assert_series_equal(y1, y2)

    def test_labels_carry_signal(self):
        """Clickers should convert more than non-clickers — the label must not be noise."""
        from app.features import make_purchase_labels, make_sample_dataframe

        df = make_sample_dataframe(n=2000, seed=5)
        y = make_purchase_labels(df, seed=5)
        high_click = y[df["click_count"] >= df["click_count"].median()].mean()
        low_click = y[df["click_count"] < df["click_count"].median()].mean()
        assert high_click > low_click

    @pytest.mark.parametrize("noise", [0.1, 0.35, 0.8])
    def test_various_noise_levels(self, sample_df, noise):
        from app.features import make_purchase_labels

        y = make_purchase_labels(sample_df, noise=noise)
        assert len(y) == len(sample_df)
