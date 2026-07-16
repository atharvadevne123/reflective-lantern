"""Tests for the hotel feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    FEATURE_COLS,
    HotelFeatureEngineer,
    build_feature_pipeline,
)


def _minimal_df(**overrides) -> pd.DataFrame:
    """Return a one-row DataFrame with valid defaults, optionally overridden."""
    defaults = {
        "lead_time": 14,
        "length_of_stay": 3,
        "guests_count": 2,
        "checkin_month": 7,
        "checkin_dayofweek": 4,
        "is_weekend": 1,
        "current_occ_rate": 0.75,
        "prev_year_occ_rate": 0.70,
        "room_rate": 189.0,
        "competitor_avg_rate": 175.0,
        "special_event": 0,
        "room_type": "deluxe",
        "booking_channel": "online",
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestHotelFeatureEngineer:
    def test_fit_returns_self(self):
        eng = HotelFeatureEngineer()
        result = eng.fit(_minimal_df())
        assert result is eng

    def test_output_columns_match_feature_cols(self):
        df = HotelFeatureEngineer().transform(_minimal_df())
        assert list(df.columns) == FEATURE_COLS

    def test_output_is_float_dtype(self):
        df = HotelFeatureEngineer().transform(_minimal_df())
        assert all(np.issubdtype(dt, np.floating) for dt in df.dtypes)

    def test_lead_time_bucket_last_minute(self):
        df = HotelFeatureEngineer().transform(_minimal_df(lead_time=1))
        assert df["lead_time_bucket"].iloc[0] == 0

    def test_lead_time_bucket_short(self):
        df = HotelFeatureEngineer().transform(_minimal_df(lead_time=10))
        assert df["lead_time_bucket"].iloc[0] == 1

    def test_lead_time_bucket_medium(self):
        df = HotelFeatureEngineer().transform(_minimal_df(lead_time=30))
        assert df["lead_time_bucket"].iloc[0] == 2

    def test_lead_time_bucket_early(self):
        df = HotelFeatureEngineer().transform(_minimal_df(lead_time=90))
        assert df["lead_time_bucket"].iloc[0] == 3

    def test_seasonality_score_summer(self):
        df = HotelFeatureEngineer().transform(_minimal_df(checkin_month=7))
        assert df["seasonality_score"].iloc[0] == pytest.approx(1.4)

    def test_seasonality_score_winter(self):
        df = HotelFeatureEngineer().transform(_minimal_df(checkin_month=1))
        assert df["seasonality_score"].iloc[0] == pytest.approx(0.7)

    def test_seasonality_score_spring(self):
        df = HotelFeatureEngineer().transform(_minimal_df(checkin_month=4))
        assert df["seasonality_score"].iloc[0] == pytest.approx(1.0)

    def test_competitor_rate_ratio_clipped(self):
        # rate = 500, competitor = 100 → ratio 5 but should clip to 3
        df = HotelFeatureEngineer().transform(
            _minimal_df(room_rate=500.0, competitor_avg_rate=100.0)
        )
        assert df["competitor_rate_ratio"].iloc[0] == pytest.approx(3.0)

    def test_competitor_rate_ratio_zero_safe(self):
        # competitor_avg_rate = 0 should not raise division error
        df = HotelFeatureEngineer().transform(_minimal_df(competitor_avg_rate=0.0, room_rate=150.0))
        assert np.isfinite(df["competitor_rate_ratio"].iloc[0])

    def test_weekend_summer_flag_true(self):
        df = HotelFeatureEngineer().transform(_minimal_df(is_weekend=1, checkin_month=7))
        assert df["weekend_summer_flag"].iloc[0] == 1

    def test_weekend_summer_flag_false_in_winter(self):
        df = HotelFeatureEngineer().transform(_minimal_df(is_weekend=1, checkin_month=1))
        assert df["weekend_summer_flag"].iloc[0] == 0

    def test_yoy_occ_delta_positive(self):
        df = HotelFeatureEngineer().transform(
            _minimal_df(current_occ_rate=0.80, prev_year_occ_rate=0.60)
        )
        assert df["yoy_occ_delta"].iloc[0] == pytest.approx(0.20)

    def test_yoy_occ_delta_clipped(self):
        df = HotelFeatureEngineer().transform(
            _minimal_df(current_occ_rate=1.0, prev_year_occ_rate=0.0)
        )
        assert df["yoy_occ_delta"].iloc[0] == pytest.approx(1.0)
