"""Tests for app.energy_intensity module."""

from __future__ import annotations

import pytest

from app.energy_intensity import (
    EUIResult,
    SECTOR_BENCHMARKS_KWH_PER_M2,
    SectorRanking,
    benchmark_eui,
    energy_use_intensity,
    normalise_weather,
    rank_within_sector,
    savings_potential,
)


class TestEnergyUseIntensity:
    def test_basic(self) -> None:
        eui = energy_use_intensity(18000.0, 100.0)
        assert eui == pytest.approx(180.0)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_use_intensity(18000.0, 0.0)

    def test_negative_energy_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_use_intensity(-100.0, 100.0)

    def test_zero_energy_ok(self) -> None:
        assert energy_use_intensity(0.0, 100.0) == 0.0

    @pytest.mark.parametrize("kwh,area,expected", [
        (10000.0, 100.0, 100.0),
        (36000.0, 200.0, 180.0),
    ])
    def test_parametrized(self, kwh: float, area: float, expected: float) -> None:
        assert energy_use_intensity(kwh, area) == pytest.approx(expected)


class TestNormaliseWeather:
    def test_no_change_when_same_hdd(self) -> None:
        eui = normalise_weather(180.0, 2000.0, 2000.0)
        assert eui == pytest.approx(180.0)

    def test_warmer_year_increases_normalised_eui(self) -> None:
        eui = normalise_weather(180.0, 1500.0, 2000.0)
        assert eui > 180.0

    def test_zero_eui_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_weather(0.0, 2000.0, 2000.0)

    def test_zero_actual_hdd_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_weather(180.0, 0.0, 2000.0)

    def test_zero_baseline_hdd_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_weather(180.0, 2000.0, 0.0)


class TestBenchmarkEUI:
    def test_office_sector(self) -> None:
        result = benchmark_eui(18000.0, 100.0, "office")
        assert isinstance(result, EUIResult)
        assert result.sector == "office"

    def test_unknown_sector_raises(self) -> None:
        with pytest.raises(ValueError):
            benchmark_eui(18000.0, 100.0, "moon_base")

    def test_rating_a_plus_for_very_efficient(self) -> None:
        benchmark = SECTOR_BENCHMARKS_KWH_PER_M2["office"]
        result = benchmark_eui(benchmark * 0.3 * 100, 100.0, "office")
        assert result.rating in ("A+", "A")

    def test_rating_f_for_very_inefficient(self) -> None:
        benchmark = SECTOR_BENCHMARKS_KWH_PER_M2["office"]
        result = benchmark_eui(benchmark * 2.0 * 100, 100.0, "office")
        assert result.rating in ("E", "F")

    def test_at_benchmark_is_rating_c_or_better(self) -> None:
        benchmark = SECTOR_BENCHMARKS_KWH_PER_M2["hotel"]
        result = benchmark_eui(benchmark * 100.0, 100.0, "hotel")
        assert result.performance_ratio == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize("sector", list(SECTOR_BENCHMARKS_KWH_PER_M2.keys()))
    def test_all_sectors_work(self, sector: str) -> None:
        result = benchmark_eui(10000.0, 100.0, sector)
        assert result.sector == sector


class TestRankWithinSector:
    def test_best_building_gets_high_percentile(self) -> None:
        peers = [200.0, 180.0, 150.0, 130.0, 100.0]
        result = rank_within_sector("b1", 100.0, peers, "office")
        assert result.percentile == pytest.approx(80.0)

    def test_worst_building_gets_zero_percentile(self) -> None:
        peers = [100.0, 150.0, 180.0, 200.0, 250.0]
        result = rank_within_sector("b1", 250.0, peers, "office")
        assert result.percentile == pytest.approx(0.0)

    def test_empty_peers_raises(self) -> None:
        with pytest.raises(ValueError):
            rank_within_sector("b1", 150.0, [], "office")

    def test_zero_eui_raises(self) -> None:
        with pytest.raises(ValueError):
            rank_within_sector("b1", 0.0, [100.0, 200.0], "office")

    def test_energy_star_eligible_above_75(self) -> None:
        peers = [300.0, 280.0, 260.0, 100.0]
        result = rank_within_sector("b1", 100.0, peers, "office")
        assert result.energy_star_eligible


class TestSavingsPotential:
    def test_basic(self) -> None:
        result = savings_potential(200.0, 150.0, 500.0)
        assert result["energy_kwh"] == pytest.approx(25000.0)

    def test_target_above_current_raises(self) -> None:
        with pytest.raises(ValueError):
            savings_potential(150.0, 200.0, 500.0)

    def test_equal_eui_raises(self) -> None:
        with pytest.raises(ValueError):
            savings_potential(150.0, 150.0, 500.0)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            savings_potential(200.0, 150.0, 0.0)

    def test_co2_and_cost_positive(self) -> None:
        result = savings_potential(200.0, 150.0, 500.0)
        assert result["co2_kg"] > 0
        assert result["cost_dollars"] > 0
