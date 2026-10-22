"""Tests for app.flexibility_score module."""

from __future__ import annotations

import pytest

from app.flexibility_score import (
    FlexibilityAssessment,
    assess_flexibility,
    load_variability_score,
    overall_flexibility_score,
    peak_contribution_score,
    response_readiness_score,
)


class TestLoadVariabilityScore:
    def test_flat_load_low_score(self) -> None:
        score = load_variability_score([5.0] * 24)
        assert score == pytest.approx(0.0)

    def test_zero_mean_returns_zero(self) -> None:
        assert load_variability_score([0.0] * 24) == 0.0

    def test_high_variability_score_positive(self) -> None:
        vals = [0.0] * 12 + [100.0] * 12
        score = load_variability_score(vals)
        assert score > 0.0

    def test_score_within_bounds(self) -> None:
        vals = [i * 1.0 for i in range(24)]
        score = load_variability_score(vals)
        assert 0.0 <= score <= 100.0

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            load_variability_score([1.0])

    @pytest.mark.parametrize(
        "values",
        [[1.0, 1.0], [0.0, 10.0, 0.0, 10.0], [5.0] * 10],
    )
    def test_returns_non_negative(self, values: list[float]) -> None:
        assert load_variability_score(values) >= 0.0


class TestResponseReadinessScore:
    def test_instant_response_full_score(self) -> None:
        assert response_readiness_score(0) == pytest.approx(100.0)

    def test_sixty_min_zero_score(self) -> None:
        assert response_readiness_score(60) == pytest.approx(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            response_readiness_score(-1)

    def test_thirty_min_fifty(self) -> None:
        score = response_readiness_score(30)
        assert score == pytest.approx(50.0)

    @pytest.mark.parametrize("minutes", [0, 10, 30, 60, 120])
    def test_score_non_negative(self, minutes: int) -> None:
        assert response_readiness_score(minutes) >= 0.0


class TestPeakContributionScore:
    def test_load_only_at_peak_hour(self) -> None:
        profile = [0.0] * 24
        for h in range(16, 21):
            profile[h] = 10.0
        score = peak_contribution_score(profile, system_peak_hour=18)
        assert score > 0.0

    def test_zero_consumption_returns_zero(self) -> None:
        assert peak_contribution_score([0.0] * 24) == 0.0

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            peak_contribution_score([1.0] * 12)

    def test_score_within_bounds(self) -> None:
        profile = [1.0] * 24
        score = peak_contribution_score(profile)
        assert 0.0 <= score <= 100.0

    @pytest.mark.parametrize("peak_hour", [8, 12, 18, 21])
    def test_parametrized_peak_hours(self, peak_hour: int) -> None:
        score = peak_contribution_score([1.0] * 24, system_peak_hour=peak_hour)
        assert 0.0 <= score <= 100.0


class TestOverallFlexibilityScore:
    def test_equal_weights_midpoint(self) -> None:
        score = overall_flexibility_score(50.0, 50.0, 50.0)
        assert score == pytest.approx(50.0)

    def test_bad_weights_raise(self) -> None:
        with pytest.raises(ValueError):
            overall_flexibility_score(50.0, 50.0, 50.0, weights=(0.5, 0.5, 0.5))

    def test_all_max_gives_hundred(self) -> None:
        score = overall_flexibility_score(100.0, 100.0, 100.0)
        assert score == pytest.approx(100.0)

    def test_all_min_gives_zero(self) -> None:
        score = overall_flexibility_score(0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "v,r,p",
        [(80.0, 60.0, 70.0), (20.0, 30.0, 50.0), (90.0, 90.0, 90.0)],
    )
    def test_score_within_bounds(self, v: float, r: float, p: float) -> None:
        score = overall_flexibility_score(v, r, p)
        assert 0.0 <= score <= 100.0


class TestAssessFlexibility:
    def _profile(self) -> list[float]:
        return [1.0 if h < 12 else 3.0 for h in range(24)]

    def test_returns_dataclass(self) -> None:
        result = assess_flexibility("building-1", self._profile(), 5, 10.0, 3.0)
        assert isinstance(result, FlexibilityAssessment)

    def test_asset_id_preserved(self) -> None:
        result = assess_flexibility("bldg-42", self._profile(), 10, 5.0, 2.0)
        assert result.asset_id == "bldg-42"

    def test_score_within_bounds(self) -> None:
        result = assess_flexibility("x", self._profile(), 15, 8.0, 4.0)
        assert 0.0 <= result.flexibility_score <= 100.0

    def test_recommendation_is_string(self) -> None:
        result = assess_flexibility("y", self._profile(), 0, 20.0, 10.0)
        assert isinstance(result.recommendation, str) and len(result.recommendation) > 0

    @pytest.mark.parametrize("response_minutes", [0, 15, 60])
    def test_parametrized_response_times(self, response_minutes: int) -> None:
        result = assess_flexibility("z", self._profile(), response_minutes, 5.0, 2.0)
        assert 0.0 <= result.flexibility_score <= 100.0


class TestFlexibilityScoreEdgeCases:
    def _profile(self) -> list[float]:
        return [float(i % 10 + 5) for i in range(24)]

    def test_variability_constant_series_zero(self) -> None:
        assert load_variability_score([10.0] * 24) == pytest.approx(0.0)

    def test_peak_contribution_flat_load(self) -> None:
        score = peak_contribution_score([5.0] * 24)
        assert 0.0 <= score <= 100.0

    def test_response_readiness_instant_max(self) -> None:
        score = response_readiness_score(0)
        assert score == pytest.approx(100.0, rel=1e-2)

    def test_response_readiness_long_low(self) -> None:
        score_fast = response_readiness_score(1)
        score_slow = response_readiness_score(120)
        assert score_fast > score_slow

    def test_overall_score_bounded(self) -> None:
        score = overall_flexibility_score(50.0, 40.0, 60.0)
        assert 0.0 <= score <= 100.0

    def test_assess_flexibility_high_capacity(self) -> None:
        result = assess_flexibility("big", self._profile(), 5, 50.0, 25.0)
        assert result.flexibility_score > 0.0

    def test_assess_flexibility_no_capacity_low_score(self) -> None:
        result = assess_flexibility("small", self._profile(), 60, 0.0, 0.0)
        assert result.flexibility_score >= 0.0

    @pytest.mark.parametrize("dispatchable,sheddable", [(5.0, 2.0), (20.0, 10.0)])
    def test_higher_capacity_higher_or_equal_score(
        self, dispatchable: float, sheddable: float
    ) -> None:
        r_low = assess_flexibility("a", self._profile(), 15, 1.0, 0.5)
        r_high = assess_flexibility("b", self._profile(), 15, dispatchable, sheddable)
        assert r_high.flexibility_score >= r_low.flexibility_score
