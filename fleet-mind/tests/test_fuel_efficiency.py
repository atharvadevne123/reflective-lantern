"""Tests for fleet-mind app/fuel_efficiency.py."""

from __future__ import annotations

import pytest

from app.fuel_efficiency import (
    FuelMetrics,
    co2_kg_from_litres,
    efficiency_score,
    fleet_average_efficiency,
    km_per_litre,
    litres_per_100km,
    load_adjusted_consumption,
)


class TestLitresPer100km:
    def test_basic(self) -> None:
        assert litres_per_100km(100.0, 10.0) == pytest.approx(10.0)

    def test_zero_distance_returns_zero(self) -> None:
        assert litres_per_100km(0.0, 10.0) == pytest.approx(0.0)

    def test_negative_distance_returns_zero(self) -> None:
        assert litres_per_100km(-1.0, 5.0) == pytest.approx(0.0)


class TestKmPerLitre:
    def test_basic(self) -> None:
        assert km_per_litre(100.0, 10.0) == pytest.approx(10.0)

    def test_zero_fuel_returns_zero(self) -> None:
        assert km_per_litre(100.0, 0.0) == pytest.approx(0.0)


class TestEfficiencyScore:
    def test_at_baseline_returns_one(self) -> None:
        assert efficiency_score(10.0, baseline=10.0) == pytest.approx(1.0)

    def test_zero_baseline_returns_zero(self) -> None:
        assert efficiency_score(10.0, baseline=0.0) == pytest.approx(0.0)

    def test_very_high_consumption_clamps_to_zero(self) -> None:
        assert efficiency_score(30.0, baseline=10.0) == pytest.approx(0.0)

    def test_low_consumption_clamps_to_one(self) -> None:
        assert efficiency_score(1.0, baseline=10.0) == pytest.approx(1.0)


class TestLoadAdjustedConsumption:
    def test_zero_load_returns_base(self) -> None:
        assert load_adjusted_consumption(8.0, 0.0) == pytest.approx(8.0)

    def test_load_increases_consumption(self) -> None:
        result = load_adjusted_consumption(8.0, 500.0)
        assert result > 8.0


class TestFleetAverageEfficiency:
    def test_empty_returns_zero(self) -> None:
        assert fleet_average_efficiency([]) == pytest.approx(0.0)

    def test_single_vehicle(self) -> None:
        m = FuelMetrics("v1", 200.0, 20.0)
        assert fleet_average_efficiency([m]) == pytest.approx(10.0)


class TestCo2KgFromLitres:
    def test_zero_litres(self) -> None:
        assert co2_kg_from_litres(0.0) == pytest.approx(0.0)

    def test_default_factor(self) -> None:
        assert co2_kg_from_litres(10.0) == pytest.approx(23.1)
