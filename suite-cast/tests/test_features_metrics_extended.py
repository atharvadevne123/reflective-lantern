"""Extended tests for suite-cast app/features.py hotel metrics functions."""

from __future__ import annotations

import pytest

from app.features import (
    average_daily_rate,
    competitor_rate_ratio,
    lead_time_bucket_label,
    length_of_stay_bucket,
    occupancy_yoy_delta,
    revpar,
    seasonality_score_for_month,
)


class TestLeadTimeBucketLabel:
    def test_same_day_label(self) -> None:
        label = lead_time_bucket_label(0)
        assert isinstance(label, str)
        assert len(label) > 0

    def test_far_advance_label(self) -> None:
        label = lead_time_bucket_label(365)
        assert isinstance(label, str)

    def test_different_ranges_give_different_labels(self) -> None:
        near = lead_time_bucket_label(1)
        far = lead_time_bucket_label(200)
        assert near != far


class TestSeasonalityScoreForMonth:
    def test_returns_float(self) -> None:
        assert isinstance(seasonality_score_for_month(6), float)

    def test_in_unit_interval(self) -> None:
        for month in range(1, 13):
            score = seasonality_score_for_month(month)
            assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("month", list(range(1, 13)))
    def test_all_months_valid(self, month: int) -> None:
        score = seasonality_score_for_month(month)
        assert isinstance(score, float)


class TestAverageDailyRate:
    def test_basic_calculation(self) -> None:
        adr = average_daily_rate(1000.0, 10)
        assert adr == pytest.approx(100.0)

    def test_zero_rooms_returns_zero_or_raises(self) -> None:
        try:
            result = average_daily_rate(1000.0, 0)
            assert result == 0.0 or result != result  # 0 or NaN
        except (ZeroDivisionError, ValueError):
            pass


class TestRevpar:
    def test_basic_calculation(self) -> None:
        result = revpar(revenue=5000.0, available_rooms=100)
        assert result == pytest.approx(50.0)

    def test_full_occupancy(self) -> None:
        result = revpar(revenue=10_000.0, available_rooms=100)
        assert result == pytest.approx(100.0)


class TestCompetitorRateRatio:
    def test_equal_rates(self) -> None:
        ratio = competitor_rate_ratio(100.0, 100.0)
        assert ratio == pytest.approx(1.0)

    def test_above_competitor(self) -> None:
        ratio = competitor_rate_ratio(150.0, 100.0)
        assert ratio > 1.0

    def test_below_competitor(self) -> None:
        ratio = competitor_rate_ratio(80.0, 100.0)
        assert ratio < 1.0


class TestOccupancyYoyDelta:
    def test_positive_growth(self) -> None:
        delta = occupancy_yoy_delta(0.9, 0.7)
        assert delta > 0

    def test_negative_growth(self) -> None:
        delta = occupancy_yoy_delta(0.5, 0.8)
        assert delta < 0

    def test_flat_zero(self) -> None:
        assert occupancy_yoy_delta(0.7, 0.7) == pytest.approx(0.0)


class TestLengthOfStayBucket:
    def test_one_night(self) -> None:
        bucket = length_of_stay_bucket(1)
        assert isinstance(bucket, str)

    def test_long_stay(self) -> None:
        bucket = length_of_stay_bucket(30)
        assert isinstance(bucket, str)

    def test_different_lengths_differ(self) -> None:
        assert length_of_stay_bucket(1) != length_of_stay_bucket(14)
