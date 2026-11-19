"""Tests for fleet-mind app/fleet_analytics.py."""

from __future__ import annotations

import pytest

from app.fleet_analytics import (
    VehicleSummary,
    cost_per_km,
    efficiency_distribution,
    fleet_health_index,
    fleet_uptime,
    top_anomaly_vehicles,
    total_maintenance_cost,
)


def _summary(
    vid: str,
    uptime: float = 0.9,
    cost: float = 1000.0,
    anomalies: int = 2,
    km: float = 50_000.0,
    efficiency: float = 9.0,
) -> VehicleSummary:
    return VehicleSummary(vid, km, cost, uptime, anomalies, efficiency)


class TestFleetUptime:
    def test_empty_returns_zero(self) -> None:
        assert fleet_uptime([]) == pytest.approx(0.0)

    def test_mean_uptime(self) -> None:
        summaries = [_summary("v1", uptime=0.8), _summary("v2", uptime=1.0)]
        assert fleet_uptime(summaries) == pytest.approx(0.9)


class TestTotalMaintenanceCost:
    def test_sums_costs(self) -> None:
        summaries = [_summary("v1", cost=500.0), _summary("v2", cost=750.0)]
        assert total_maintenance_cost(summaries) == pytest.approx(1250.0)


class TestCostPerKm:
    def test_zero_km_returns_zero(self) -> None:
        summaries = [_summary("v1", km=0.0, cost=1000.0)]
        assert cost_per_km(summaries) == pytest.approx(0.0)

    def test_calculates_correctly(self) -> None:
        summaries = [_summary("v1", km=100_000.0, cost=1000.0)]
        assert cost_per_km(summaries) == pytest.approx(0.01)


class TestTopAnomalyVehicles:
    def test_returns_n_vehicles(self) -> None:
        summaries = [_summary(f"v{i}", anomalies=i) for i in range(10)]
        top = top_anomaly_vehicles(summaries, n=3)
        assert len(top) == 3

    def test_highest_anomaly_first(self) -> None:
        summaries = [_summary("v1", anomalies=1), _summary("v2", anomalies=5)]
        top = top_anomaly_vehicles(summaries, n=1)
        assert top[0].vehicle_id == "v2"


class TestEfficiencyDistribution:
    def test_buckets_vehicles(self) -> None:
        summaries = [
            _summary("v1", efficiency=5.0),
            _summary("v2", efficiency=7.5),
            _summary("v3", efficiency=10.5),
            _summary("v4", efficiency=15.0),
        ]
        dist = efficiency_distribution(summaries)
        assert dist["<6"] == 1
        assert dist["6-9"] == 1
        assert dist["9-12"] == 1
        assert dist[">12"] == 1


class TestFleetHealthIndex:
    def test_empty_returns_zero(self) -> None:
        assert fleet_health_index([]) == pytest.approx(0.0)

    def test_returns_between_zero_and_one(self) -> None:
        summaries = [_summary("v1"), _summary("v2")]
        idx = fleet_health_index(summaries)
        assert 0.0 <= idx <= 1.0
