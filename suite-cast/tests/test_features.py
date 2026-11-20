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

    def test_advance_efficiency_increases_with_lead_time(self):
        df_short = HotelFeatureEngineer().transform(_minimal_df(lead_time=1, checkin_month=7))
        df_long = HotelFeatureEngineer().transform(_minimal_df(lead_time=180, checkin_month=7))
        assert df_long["advance_efficiency"].iloc[0] > df_short["advance_efficiency"].iloc[0]

    @pytest.mark.parametrize(
        "room_type,expected_code",
        [
            ("standard", 0),
            ("deluxe", 1),
            ("suite", 2),
        ],
    )
    def test_room_type_encoding(self, room_type, expected_code):
        df = HotelFeatureEngineer().transform(_minimal_df(room_type=room_type))
        assert df["room_type_enc"].iloc[0] == expected_code

    @pytest.mark.parametrize(
        "channel,expected_code",
        [
            ("direct", 0),
            ("online", 1),
            ("ota", 2),
        ],
    )
    def test_channel_encoding(self, channel, expected_code):
        df = HotelFeatureEngineer().transform(_minimal_df(booking_channel=channel))
        assert df["channel_enc"].iloc[0] == expected_code

    def test_batch_transform_preserves_row_count(self):
        batch = pd.concat([_minimal_df() for _ in range(5)], ignore_index=True)
        df = HotelFeatureEngineer().transform(batch)
        assert len(df) == 5


class TestBuildFeaturePipeline:
    def test_pipeline_has_engineer_step(self):
        pipe = build_feature_pipeline()
        assert "engineer" in pipe.named_steps

    def test_pipeline_has_scaler_step(self):
        pipe = build_feature_pipeline()
        assert "scaler" in pipe.named_steps

    def test_pipeline_fit_transform_returns_array(self):
        from app.model import generate_sample_data

        X, _ = generate_sample_data(50)
        pipe = build_feature_pipeline()
        result = pipe.fit_transform(X)
        assert result.shape[0] == 50
        assert result.shape[1] == len(FEATURE_COLS)


class TestLeadTimeBucket:
    @pytest.mark.parametrize(
        "lead_days,expected",
        [
            (0, 0),
            (3, 0),
            (4, 1),
            (14, 1),
            (15, 2),
            (60, 2),
            (61, 3),
            (365, 3),
            (-5, 0),
            (400, 3),
        ],
    )
    def test_bucket_boundaries(self, lead_days, expected):
        from app.features import lead_time_bucket

        assert lead_time_bucket(lead_days) == expected


class TestCompetitorRateRatio:
    def test_ratio_equal_rates(self):
        from app.features import competitor_rate_ratio

        assert competitor_rate_ratio(100.0, 100.0) == pytest.approx(1.0)

    def test_zero_competitor_uses_fallback(self):
        from app.features import competitor_rate_ratio

        result = competitor_rate_ratio(150.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_clamped_above(self):
        from app.features import competitor_rate_ratio

        assert competitor_rate_ratio(9999.0, 1.0) == pytest.approx(3.0)

    def test_clamped_below(self):
        from app.features import competitor_rate_ratio

        assert competitor_rate_ratio(1.0, 9999.0) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "room,comp,expected",
        [
            (200.0, 200.0, 1.0),
            (300.0, 200.0, 1.5),
            (100.0, 200.0, 0.5),
        ],
    )
    def test_parametrized(self, room, comp, expected):
        from app.features import competitor_rate_ratio

        assert competitor_rate_ratio(room, comp) == pytest.approx(expected)


class TestOccupancyYoyDelta:
    def test_no_change(self):
        from app.features import occupancy_yoy_delta

        assert occupancy_yoy_delta(0.8, 0.8) == pytest.approx(0.0)

    def test_positive_delta(self):
        from app.features import occupancy_yoy_delta

        assert occupancy_yoy_delta(0.9, 0.7) == pytest.approx(0.2)

    def test_clamped_to_minus_one(self):
        from app.features import occupancy_yoy_delta

        assert occupancy_yoy_delta(0.0, 1.0) == pytest.approx(-1.0)

    def test_clamped_to_one(self):
        from app.features import occupancy_yoy_delta

        assert occupancy_yoy_delta(1.0, 0.0) == pytest.approx(1.0)


class TestAdrFromRevenue:
    def test_basic(self) -> None:
        from app.features import adr_from_revenue

        assert adr_from_revenue(1000.0, 10) == pytest.approx(100.0, abs=0.01)

    def test_zero_rooms_returns_zero(self) -> None:
        from app.features import adr_from_revenue

        assert adr_from_revenue(5000.0, 0) == 0.0

    def test_negative_rooms_returns_zero(self) -> None:
        from app.features import adr_from_revenue

        assert adr_from_revenue(5000.0, -5) == 0.0


class TestRevpar:
    def test_basic(self) -> None:
        from app.features import revpar

        assert revpar(800.0, 10) == pytest.approx(80.0, abs=0.01)

    def test_zero_available_returns_zero(self) -> None:
        from app.features import revpar

        assert revpar(1000.0, 0) == 0.0

    def test_full_occupancy_equals_adr(self) -> None:
        from app.features import adr_from_revenue, revpar

        rev, rooms = 500.0, 5
        assert revpar(rev, rooms) == pytest.approx(adr_from_revenue(rev, rooms), abs=0.01)


class TestLengthOfStayBucket:
    def test_short(self) -> None:
        from app.features import length_of_stay_bucket

        assert length_of_stay_bucket(1) == "short"
        assert length_of_stay_bucket(2) == "short"

    def test_medium(self) -> None:
        from app.features import length_of_stay_bucket

        assert length_of_stay_bucket(3) == "medium"
        assert length_of_stay_bucket(6) == "medium"

    def test_long(self) -> None:
        from app.features import length_of_stay_bucket

        assert length_of_stay_bucket(7) == "long"
        assert length_of_stay_bucket(30) == "long"


@pytest.mark.parametrize(
    "month,expected_season",
    [
        (1, "winter"),
        (4, "spring"),
        (7, "summer"),
        (10, "fall"),
        (12, "winter"),
    ],
)
def test_seasonality_score_by_month(month: int, expected_season: str) -> None:
    from app.features import _MONTH_TO_SEASON

    assert _MONTH_TO_SEASON[month] == expected_season


@pytest.mark.parametrize("lead_time", [0, 1, 3, 7, 14, 30, 90])
def test_feature_engineer_handles_various_lead_times(lead_time: int) -> None:
    eng = HotelFeatureEngineer()
    df = _minimal_df(lead_time=lead_time)
    result = eng.transform(df)
    assert result.shape[0] == 1
    assert not result.isnull().any().any()


@pytest.mark.parametrize("room_type", ["standard", "deluxe", "suite"])
def test_feature_engineer_all_room_types(room_type: str) -> None:
    eng = HotelFeatureEngineer()
    result = eng.transform(_minimal_df(room_type=room_type))
    assert result.shape == (1, len(FEATURE_COLS))


@pytest.mark.parametrize("guests_count", [1, 2, 4, 8])
def test_feature_engineer_various_guest_counts(guests_count: int) -> None:
    eng = HotelFeatureEngineer()
    result = eng.transform(_minimal_df(guests_count=guests_count))
    assert not result.isnull().any().any()


class TestRevenuePerAvailableRoom:
    def test_basic_revpar(self) -> None:
        from app.features import revenue_per_available_room

        assert revenue_per_available_room(1000.0, 10) == pytest.approx(100.0)

    def test_zero_rooms_returns_zero(self) -> None:
        from app.features import revenue_per_available_room

        assert revenue_per_available_room(500.0, 0) == 0.0

    @pytest.mark.parametrize("rooms,expected", [(1, 100.0), (10, 10.0), (100, 1.0)])
    def test_revpar_scales_with_rooms(self, rooms: int, expected: float) -> None:
        from app.features import revenue_per_available_room

        assert revenue_per_available_room(100.0, rooms) == pytest.approx(expected)


class TestBookingConversionRate:
    def test_full_conversion(self) -> None:
        from app.features import booking_conversion_rate

        assert booking_conversion_rate(10, 10) == pytest.approx(1.0)

    def test_zero_enquiries_returns_zero(self) -> None:
        from app.features import booking_conversion_rate

        assert booking_conversion_rate(0, 0) == 0.0

    def test_capped_at_one(self) -> None:
        from app.features import booking_conversion_rate

        assert booking_conversion_rate(5, 10) == pytest.approx(1.0)

    @pytest.mark.parametrize("enquiries,confirmed,expected", [(10, 5, 0.5), (100, 25, 0.25)])
    def test_partial_conversion(self, enquiries: int, confirmed: int, expected: float) -> None:
        from app.features import booking_conversion_rate

        assert booking_conversion_rate(enquiries, confirmed) == pytest.approx(expected)


class TestAverageDailyRate:
    def test_basic_adr(self) -> None:
        from app.features import average_daily_rate

        assert average_daily_rate(1000.0, 10) == pytest.approx(100.0)

    def test_zero_occupied_returns_zero(self) -> None:
        from app.features import average_daily_rate

        assert average_daily_rate(500.0, 0) == 0.0


class TestGrossOperatingProfitPerAvailableRoom:
    def test_standard_case(self) -> None:
        from app.features import gross_operating_profit_per_available_room

        result = gross_operating_profit_per_available_room(50000.0, 30000.0, 100)
        assert result == pytest.approx(200.0)

    def test_zero_rooms_returns_zero(self) -> None:
        from app.features import gross_operating_profit_per_available_room

        assert gross_operating_profit_per_available_room(50000.0, 30000.0, 0) == 0.0

    def test_loss_is_negative(self) -> None:
        from app.features import gross_operating_profit_per_available_room

        result = gross_operating_profit_per_available_room(10000.0, 30000.0, 100)
        assert result < 0.0


class TestLengthOfStayMix:
    def test_empty_returns_zeros(self) -> None:
        from app.features import length_of_stay_mix

        result = length_of_stay_mix([])
        assert all(v == 0.0 for v in result.values())

    def test_fractions_sum_to_one(self) -> None:
        from app.features import length_of_stay_mix

        result = length_of_stay_mix([1, 2, 3, 5])
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-4)

    def test_only_one_night_stays(self) -> None:
        from app.features import length_of_stay_mix

        result = length_of_stay_mix([1, 1, 1])
        assert result["one_night"] == 1.0
        assert result["two_three_nights"] == 0.0

    def test_keys_present(self) -> None:
        from app.features import length_of_stay_mix

        result = length_of_stay_mix([2])
        assert {"one_night", "two_three_nights", "four_plus_nights"} <= result.keys()
