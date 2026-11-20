"""Tests for fleet-mind app/risk_scorer.py."""

from __future__ import annotations

from app.risk_scorer import (
    RiskInput,
    component_scores,
    composite_risk_score,
    fleet_risk_summary,
    risk_tier,
)


def _make_input(**kwargs) -> RiskInput:
    defaults = dict(
        vehicle_id="v1",
        anomaly_score=0.5,
        days_maintenance_overdue=30,
        vehicle_age_years=5.0,
        odometer_km=100_000.0,
    )
    defaults.update(kwargs)
    return RiskInput(**defaults)


class TestComponentScores:
    def test_returns_expected_keys(self) -> None:
        scores = component_scores(_make_input())
        assert "anomaly" in scores
        assert "overdue_maintenance" in scores
        assert "age" in scores
        assert "mileage" in scores

    def test_all_between_zero_and_one(self) -> None:
        scores = component_scores(_make_input())
        for v in scores.values():
            assert 0.0 <= v <= 1.0


class TestCompositeRiskScore:
    def test_returns_float_between_zero_and_one(self) -> None:
        score = composite_risk_score(_make_input())
        assert 0.0 <= score <= 1.0

    def test_high_anomaly_increases_score(self) -> None:
        low = composite_risk_score(_make_input(anomaly_score=0.1))
        high = composite_risk_score(_make_input(anomaly_score=0.9))
        assert high > low


class TestRiskTier:
    def test_low_tier(self) -> None:
        assert risk_tier(0.1) == "LOW"

    def test_medium_tier(self) -> None:
        assert risk_tier(0.35) == "MEDIUM"

    def test_high_tier(self) -> None:
        assert risk_tier(0.6) == "HIGH"

    def test_critical_tier(self) -> None:
        assert risk_tier(0.9) == "CRITICAL"


class TestFleetRiskSummary:
    def test_empty_fleet(self) -> None:
        summary = fleet_risk_summary([])
        assert summary["count"] == 0

    def test_counts_critical(self) -> None:
        inputs = [
            _make_input(vehicle_id="v1", anomaly_score=1.0, days_maintenance_overdue=180),
            _make_input(vehicle_id="v2", anomaly_score=0.0, days_maintenance_overdue=0),
        ]
        summary = fleet_risk_summary(inputs)
        assert summary["count"] == 2
        assert isinstance(summary["mean_score"], float)
