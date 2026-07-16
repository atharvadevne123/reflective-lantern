"""Tests for model training, prediction, and data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model import generate_sample_data, predict_demand


class TestGenerateSampleData:
    def test_returns_dataframe_and_array(self):
        df, y = generate_sample_data(100)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(y, np.ndarray)

    def test_row_count_matches_n(self):
        df, y = generate_sample_data(50)
        assert len(df) == 50
        assert len(y) == 50

    def test_target_is_binary(self):
        _, y = generate_sample_data(200)
        assert set(np.unique(y)).issubset({0, 1})

    def test_required_columns_present(self):
        df, _ = generate_sample_data(10)
        required = {
            "lead_time",
            "length_of_stay",
            "guests_count",
            "checkin_month",
            "checkin_dayofweek",
            "is_weekend",
            "current_occ_rate",
            "prev_year_occ_rate",
            "room_rate",
            "competitor_avg_rate",
            "special_event",
            "room_type",
            "booking_channel",
        }
        assert required.issubset(set(df.columns))

    def test_room_rate_is_positive(self):
        df, _ = generate_sample_data(50)
        assert (df["room_rate"] > 0).all()

    def test_occ_rate_in_valid_range(self):
        df, _ = generate_sample_data(100)
        assert df["current_occ_rate"].between(0, 1).all()

    @pytest.mark.parametrize("n", [10, 100, 500])
    def test_reproducible_with_fixed_seed(self, n):
        _, y1 = generate_sample_data(n)
        _, y2 = generate_sample_data(n)
        np.testing.assert_array_equal(y1, y2)


class TestTrainModel:
    def test_returns_three_items(self, trained_models):
        assert len(trained_models) == 3

    def test_metrics_has_auc_mean(self, trained_models):
        _, _, metrics = trained_models
        assert "auc_mean" in metrics
        assert 0.0 <= float(metrics["auc_mean"]) <= 1.0

    def test_auc_mean_above_random(self, trained_models):
        _, _, metrics = trained_models
        assert float(metrics["auc_mean"]) > 0.55

    def test_n_features_in_metrics(self, trained_models):
        _, _, metrics = trained_models
        assert metrics["n_features"] == 18

    def test_both_models_fit(self, trained_models):
        xgb_pipe, lgbm_pipe, _ = trained_models
        assert xgb_pipe is not None
        assert lgbm_pipe is not None

    def test_xgb_can_predict_proba(self, trained_models, sample_booking_df):
        xgb_pipe, _, _ = trained_models
        probs = xgb_pipe.predict_proba(sample_booking_df)
        assert probs.shape[1] == 2
        assert 0.0 <= probs[0, 1] <= 1.0
