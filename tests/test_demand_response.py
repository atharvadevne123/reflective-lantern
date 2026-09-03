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


class TestDemandResponseHelpers:
    def test_curtailment_rate_basic(self) -> None:
        from app.demand_response import curtailment_rate

        assert curtailment_rate(8.0, event_hours=4) == pytest.approx(2.0)

    def test_curtailment_rate_zero_hours_raises(self) -> None:
        from app.demand_response import curtailment_rate

        with pytest.raises(ValueError, match="event_hours must be positive"):
            curtailment_rate(10.0, event_hours=0)

    def test_event_roi_positive_when_profitable(self) -> None:
        from app.demand_response import event_roi

        roi = event_roi(net_payment=10.0, baseline_cost_per_kwh=0.10, baseline_kwh=50.0)
        assert roi > 0.0

    def test_event_roi_zero_baseline_returns_zero(self) -> None:
        from app.demand_response import event_roi

        assert event_roi(net_payment=10.0, baseline_cost_per_kwh=0.0, baseline_kwh=50.0) == 0.0

    def test_event_roi_negative_rate_raises(self) -> None:
        from app.demand_response import event_roi

        with pytest.raises(ValueError):
            event_roi(net_payment=10.0, baseline_cost_per_kwh=-0.1, baseline_kwh=50.0)
