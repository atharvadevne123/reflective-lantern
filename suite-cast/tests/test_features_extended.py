"""Extended tests for suite-cast app/features.py helper functions."""

from __future__ import annotations

import pytest

from app.features import (
    BOOKING_CHANNELS,
    FEATURE_COLS,
    ROOM_TYPES,
    average_daily_rate,
    booking_conversion_rate,
    competitor_rate_ratio,
    lead_time_bucket,
    lead_time_bucket_label,
    length_of_stay_mix,
    occupancy_yoy_delta,
    revpar,
    revenue_per_available_room,
    seasonality_score_for_month,
)


class TestLeadTimeBucketLabel:
    def test_last_minute(self) -> None:
        assert lead_time_bucket_label(0) == "last_minute"
        assert lead_time_bucket_label(3) == "last_minute"

    def test_short(self) -> None:
        assert lead_time_bucket_label(4) == "short"
        assert lead_time_bucket_label(14) == "short"

    def test_medium(self) -> None:
        assert lead_time_bucket_label(15) == "medium"
        assert lead_time_bucket_label(60) == "medium"

    def test_advance(self) -> None:
        assert lead_time_bucket_label(61) == "advance"
        assert lead_time_bucket_label(200) == "advance"


class TestSeasonalityScore:
    def test_summer_is_highest(self) -> None:
        july = seasonality_score_for_month(7)
        jan = seasonality_score_for_month(1)
        assert july > jan

    def test_all_months_positive(self) -> None:
        for m in range(1, 13):
            assert seasonality_score_for_month(m) > 0.0

    def test_known_winter_score(self) -> None:
        assert seasonality_score_for_month(1) == pytest.approx(0.70)

    def test_known_summer_score(self) -> None:
        assert seasonality_score_for_month(7) == pytest.approx(1.40)


class TestRevenuePerAvailableRoom:
    def test_basic_calculation(self) -> None:
        assert revenue_per_available_room(10000.0, 100) == pytest.approx(100.0)

    def test_zero_rooms_returns_zero(self) -> None:
        assert revenue_per_available_room(10000.0, 0) == pytest.approx(0.0)


class TestBookingConversionRate:
    def test_full_conversion(self) -> None:
        assert booking_conversion_rate(100, 100) == pytest.approx(1.0)

    def test_zero_enquiries(self) -> None:
        assert booking_conversion_rate(0, 10) == pytest.approx(0.0)

    def test_partial_conversion(self) -> None:
        rate = booking_conversion_rate(100, 50)
        assert rate == pytest.approx(0.5)

    def test_caps_at_one(self) -> None:
        assert booking_conversion_rate(10, 200) == pytest.approx(1.0)


class TestAverageDailyRate:
    def test_basic(self) -> None:
        assert average_daily_rate(5000.0, 50) == pytest.approx(100.0)

    def test_zero_rooms(self) -> None:
        assert average_daily_rate(5000.0, 0) == pytest.approx(0.0)


class TestCompetitorRateRatio:
    def test_basic_ratio(self) -> None:
        ratio = competitor_rate_ratio(150.0, 100.0)
        assert ratio == pytest.approx(1.5)

    def test_capped_at_three(self) -> None:
        assert competitor_rate_ratio(1000.0, 100.0) == pytest.approx(3.0)

    def test_floored_at_half(self) -> None:
        assert competitor_rate_ratio(10.0, 1000.0) == pytest.approx(0.5)

    def test_zero_competitor_uses_fallback(self) -> None:
        ratio = competitor_rate_ratio(150.0, 0.0)
        assert ratio > 0.0


class TestOccupancyYoyDelta:
    def test_positive_delta(self) -> None:
        assert occupancy_yoy_delta(0.9, 0.7) == pytest.approx(0.2)

    def test_negative_delta(self) -> None:
        assert occupancy_yoy_delta(0.7, 0.9) == pytest.approx(-0.2)

    def test_clamped_positive(self) -> None:
        assert occupancy_yoy_delta(1.0, 0.0) == pytest.approx(1.0)

    def test_clamped_negative(self) -> None:
        assert occupancy_yoy_delta(0.0, 1.0) == pytest.approx(-1.0)


class TestLengthOfStayMix:
    def test_empty_returns_zeros(self) -> None:
        mix = length_of_stay_mix([])
        assert mix == {"one_night": 0.0, "two_three_nights": 0.0, "four_plus_nights": 0.0}

    def test_all_one_night(self) -> None:
        mix = length_of_stay_mix([1, 1, 1])
        assert mix["one_night"] == pytest.approx(1.0)
        assert mix["four_plus_nights"] == pytest.approx(0.0)

    def test_mixed_stays(self) -> None:
        mix = length_of_stay_mix([1, 2, 4])
        assert sum(mix.values()) == pytest.approx(1.0)


class TestRevpar:
    def test_basic(self) -> None:
        assert revpar(5000.0, 100) == pytest.approx(50.0)

    def test_zero_rooms(self) -> None:
        assert revpar(5000.0, 0) == pytest.approx(0.0)


class TestConstants:
    def test_room_types_not_empty(self) -> None:
        assert len(ROOM_TYPES) > 0

    def test_booking_channels_not_empty(self) -> None:
        assert len(BOOKING_CHANNELS) > 0

    def test_feature_cols_not_empty(self) -> None:
        assert len(FEATURE_COLS) > 0
