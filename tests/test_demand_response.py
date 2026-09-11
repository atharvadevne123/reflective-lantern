"""Tests for app/demand_response.py."""

from __future__ import annotations

import pytest

from app.demand_response import (
    curtailment,
    customer_baseline_load,
    evaluate_event,
    performance_score,
)

BASELINE = [10.0, 10.0, 10.0, 10.0]
CURTAILED = [6.0, 6.0, 6.0, 6.0]


class TestCustomerBaselineLoad:
    def test_averages_across_days(self) -> None:
        assert customer_baseline_load([[10.0, 20.0], [20.0, 40.0]]) == [15.0, 30.0]

    def test_single_day_is_that_day(self) -> None:
        assert customer_baseline_load([[5.0, 7.0]]) == [5.0, 7.0]

    def test_uses_only_most_recent_days(self) -> None:
        history = [[100.0], [100.0], [10.0], [20.0]]
        assert customer_baseline_load(history, days=2) == [15.0]

    def test_window_larger_than_history_uses_all(self) -> None:
        assert customer_baseline_load([[10.0], [20.0]], days=99) == [15.0]

    def test_preserves_hour_count(self) -> None:
        baseline = customer_baseline_load([[1.0] * 24, [3.0] * 24])
        assert len(baseline) == 24

    def test_empty_history_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            customer_baseline_load([])

    @pytest.mark.parametrize("days", [0, -1])
    def test_non_positive_days_rejected(self, days: int) -> None:
        with pytest.raises(ValueError, match="days must be positive"):
            customer_baseline_load([[1.0]], days=days)

    def test_ragged_days_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            customer_baseline_load([[1.0, 2.0], [1.0]])


class TestCurtailment:
    def test_reduction_is_positive(self) -> None:
        assert curtailment(BASELINE, CURTAILED) == pytest.approx(16.0)

    def test_no_change_is_zero(self) -> None:
        assert curtailment(BASELINE, BASELINE) == 0.0

    def test_over_consumption_is_negative(self) -> None:
        assert curtailment(BASELINE, [12.0] * 4) == pytest.approx(-8.0)

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            curtailment(BASELINE, [6.0])


class TestPerformanceScore:
    def test_exact_delivery_scores_one(self) -> None:
        assert performance_score(10.0, 10.0) == pytest.approx(1.0)

    def test_half_delivery_scores_half(self) -> None:
        assert performance_score(5.0, 10.0) == pytest.approx(0.5)

    def test_over_delivery_caps_at_one(self) -> None:
        assert performance_score(50.0, 10.0) == 1.0

    def test_negative_curtailment_floors_at_zero(self) -> None:
        assert performance_score(-5.0, 10.0) == 0.0

    def test_zero_commitment_scores_one(self) -> None:
        assert performance_score(0.0, 0.0) == 1.0

    def test_negative_commitment_rejected(self) -> None:
        with pytest.raises(ValueError, match="committed_kwh must be non-negative"):
            performance_score(5.0, -1.0)


class TestEvaluateEvent:
    def test_full_delivery_pays_incentive_without_penalty(self) -> None:
        result = evaluate_event(BASELINE, CURTAILED, committed_kwh=16.0)
        assert result.curtailed_kwh == pytest.approx(16.0)
        assert result.shortfall_kwh == 0.0
        assert result.penalty == 0.0
        assert result.net_payment == result.incentive

    def test_shortfall_incurs_penalty(self) -> None:
        result = evaluate_event(BASELINE, [8.0] * 4, committed_kwh=16.0)
        assert result.shortfall_kwh == pytest.approx(8.0)
        assert result.penalty > 0
        assert result.net_payment < result.incentive

    def test_over_delivery_pays_full_curtailment(self) -> None:
        result = evaluate_event(BASELINE, CURTAILED, committed_kwh=5.0)
        assert result.performance_score == 1.0
        assert result.shortfall_kwh == 0.0
        assert result.incentive == pytest.approx(round(16.0 * 1.25, 2))

    def test_no_curtailment_earns_nothing(self) -> None:
        result = evaluate_event(BASELINE, BASELINE, committed_kwh=0.0)
        assert result.curtailed_kwh == 0.0
        assert result.incentive == 0.0
        assert result.net_payment == 0.0

    def test_over_consumption_earns_no_incentive(self) -> None:
        result = evaluate_event(BASELINE, [12.0] * 4, committed_kwh=0.0)
        assert result.curtailed_kwh < 0
        assert result.incentive == 0.0
        assert result.performance_score == 1.0

    def test_curtailment_pct_matches_totals(self) -> None:
        result = evaluate_event(BASELINE, CURTAILED, committed_kwh=16.0)
        assert result.curtailment_pct == pytest.approx(40.0)
        assert result.baseline_kwh == pytest.approx(40.0)
        assert result.actual_kwh == pytest.approx(24.0)

    def test_net_payment_is_incentive_minus_penalty(self) -> None:
        result = evaluate_event(BASELINE, [9.0] * 4, committed_kwh=20.0)
        assert result.net_payment == pytest.approx(round(result.incentive - result.penalty, 2))

    def test_zero_baseline_reports_zero_pct(self) -> None:
        result = evaluate_event([0.0] * 4, [0.0] * 4, committed_kwh=0.0)
        assert result.curtailment_pct == 0.0

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            evaluate_event(BASELINE, [6.0], committed_kwh=1.0)

    @pytest.mark.parametrize(
        ("incentive", "penalty"),
        [(-1.0, 0.75), (1.25, -0.5)],
    )
    def test_negative_rates_rejected(self, incentive: float, penalty: float) -> None:
        with pytest.raises(ValueError, match="rates must be non-negative"):
            evaluate_event(BASELINE, CURTAILED, 16.0, incentive_per_kwh=incentive, penalty_per_kwh=penalty)


class TestCurtailmentEdgeCases:
    @pytest.mark.parametrize(
        "baseline,actual,expected",
        [
            ([10.0, 10.0], [5.0, 5.0], 10.0),
            ([0.0, 0.0], [0.0, 0.0], 0.0),
            ([5.0], [5.0], 0.0),
        ],
    )
    def test_various_curtailment_values(self, baseline: list, actual: list, expected: float) -> None:
        assert curtailment(baseline, actual) == pytest.approx(expected)

    def test_single_hour_curtailment(self) -> None:
        assert curtailment([10.0], [3.0]) == pytest.approx(7.0)


class TestPerformanceScoreParametrize:
    @pytest.mark.parametrize(
        "curtailed,committed,expected",
        [
            (10.0, 10.0, 1.0),
            (5.0, 10.0, 0.5),
            (15.0, 10.0, 1.0),
            (0.0, 10.0, 0.0),
        ],
    )
    def test_score_values(self, curtailed: float, committed: float, expected: float) -> None:
        assert performance_score(curtailed, committed) == pytest.approx(expected)

    def test_score_is_bounded(self) -> None:
        for curtailed, committed in [(0.0, 5.0), (5.0, 5.0), (100.0, 5.0)]:
            score = performance_score(curtailed, committed)
            assert 0.0 <= score <= 1.0


class TestEvaluateEventFields:
    def test_result_has_all_required_fields(self) -> None:
        result = evaluate_event(BASELINE, CURTAILED, committed_kwh=10.0)
        for field in (
            "baseline_kwh",
            "actual_kwh",
            "curtailed_kwh",
            "curtailment_pct",
            "committed_kwh",
            "shortfall_kwh",
            "incentive",
            "penalty",
            "net_payment",
            "performance_score",
        ):
            assert hasattr(result, field)

    def test_performance_score_bounded(self) -> None:
        result = evaluate_event(BASELINE, CURTAILED, committed_kwh=16.0)
        assert 0.0 <= result.performance_score <= 1.0


@pytest.mark.parametrize("n_hours", [1, 4, 8, 24])
def test_curtailment_output_length_matches_input(n_hours: int) -> None:
    """curtailment result has same number of elements as input."""
    baseline = [5.0] * n_hours
    actual = [3.0] * n_hours
    result = curtailment(baseline, actual)
    assert len(result) == n_hours


@pytest.mark.parametrize("reduction", [0.0, 0.25, 0.5, 1.0])
def test_performance_score_for_known_fractions(reduction: float) -> None:
    """performance_score returns reduction fraction when committed > 0."""
    score = performance_score(curtailed=reduction * 10.0, committed_kwh=10.0)
    assert score == pytest.approx(min(1.0, reduction))


class TestCustomerBaselineLoadEdgeCases:
    def test_two_days_average(self) -> None:
        result = customer_baseline_load([[4.0, 8.0], [8.0, 4.0]])
        assert result == pytest.approx([6.0, 6.0])

    @pytest.mark.parametrize("n_days", [1, 3, 7])
    def test_baseline_length_equals_day_length(self, n_days: int) -> None:
        history = [[1.0] * 4] * n_days
        result = customer_baseline_load(history)
        assert len(result) == 4


from app.demand_response import curtailment_rate, event_roi


class TestCurtailmentRate:
    def test_positive_result_for_valid_inputs(self) -> None:
        rate = curtailment_rate(curtailed_kwh=20.0, event_hours=4)
        assert rate == pytest.approx(5.0)

    def test_zero_event_hours_raises(self) -> None:
        with pytest.raises((ValueError, ZeroDivisionError)):
            curtailment_rate(curtailed_kwh=20.0, event_hours=0)

    @pytest.mark.parametrize("hours", [1, 2, 4, 8])
    def test_rate_scales_inversely_with_hours(self, hours: int) -> None:
        rate = curtailment_rate(curtailed_kwh=40.0, event_hours=hours)
        assert rate == pytest.approx(40.0 / hours)


class TestEventRoi:
    def test_positive_roi_for_profitable_event(self) -> None:
        roi = event_roi(net_payment=100.0, baseline_cost_per_kwh=0.10, baseline_kwh=500.0)
        assert isinstance(roi, float)

    def test_zero_payment_gives_low_roi(self) -> None:
        roi_zero = event_roi(net_payment=0.0, baseline_cost_per_kwh=0.10, baseline_kwh=100.0)
        roi_pos = event_roi(net_payment=50.0, baseline_cost_per_kwh=0.10, baseline_kwh=100.0)
        assert roi_pos > roi_zero

    @pytest.mark.parametrize("payment", [0.0, 50.0, 200.0])
    def test_roi_is_finite_for_valid_inputs(self, payment: float) -> None:
        import math
        roi = event_roi(net_payment=payment, baseline_cost_per_kwh=0.10, baseline_kwh=100.0)
        assert math.isfinite(roi)
