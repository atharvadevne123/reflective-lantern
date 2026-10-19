"""Tests for app.peak_demand module."""

from __future__ import annotations

import pytest

from app.peak_demand import (
    PeakDemandResult,
    analyse_peak_demand,
    demand_charge,
    demand_response_savings,
    load_factor,
    peak_kw,
    ratchet_demand,
)


class TestPeakKw:
    def test_basic_maximum(self) -> None:
        assert peak_kw([10.0, 50.0, 30.0]) == 50.0

    def test_empty_list_returns_zero(self) -> None:
        assert peak_kw([]) == 0.0

    def test_negative_reading_raises(self) -> None:
        with pytest.raises(ValueError):
            peak_kw([-1.0, 10.0])

    def test_single_element(self) -> None:
        assert peak_kw([42.0]) == 42.0

    @pytest.mark.parametrize(
        "readings,expected",
        [
            ([1.0, 2.0, 3.0], 3.0),
            ([5.0, 5.0, 5.0], 5.0),
            ([0.0, 0.0], 0.0),
        ],
    )
    def test_parametrized(self, readings: list[float], expected: float) -> None:
        assert peak_kw(readings) == expected


class TestLoadFactor:
    def test_flat_load_gives_one(self) -> None:
        assert load_factor([5.0, 5.0, 5.0]) == pytest.approx(1.0)

    def test_zero_peak_gives_zero(self) -> None:
        assert load_factor([0.0, 0.0]) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            load_factor([])

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            load_factor([-1.0, 5.0])

    def test_typical_building(self) -> None:
        readings = [10.0, 20.0, 50.0, 30.0, 10.0]
        lf = load_factor(readings)
        expected = (sum(readings) / len(readings)) / max(readings)
        assert lf == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize(
        "readings",
        [[1.0, 10.0], [5.0, 5.0, 10.0, 5.0], [0.0, 5.0, 10.0]],
    )
    def test_within_zero_one(self, readings: list[float]) -> None:
        lf = load_factor(readings)
        assert 0.0 <= lf <= 1.0


class TestDemandCharge:
    def test_basic_calculation(self) -> None:
        assert demand_charge(100.0, 12.0) == pytest.approx(1200.0)

    def test_zero_peak_zero_charge(self) -> None:
        assert demand_charge(0.0) == 0.0

    def test_negative_peak_raises(self) -> None:
        with pytest.raises(ValueError):
            demand_charge(-10.0)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            demand_charge(10.0, -1.0)

    @pytest.mark.parametrize(
        "peak,rate,expected",
        [(50.0, 10.0, 500.0), (100.0, 15.0, 1500.0), (0.0, 12.0, 0.0)],
    )
    def test_parametrized(self, peak: float, rate: float, expected: float) -> None:
        assert demand_charge(peak, rate) == pytest.approx(expected)


class TestRatchetDemand:
    def test_current_exceeds_ratchet(self) -> None:
        # current=100, history=[50,60,70], ratchet=0.85*70=59.5 → bill=100
        result = ratchet_demand([50.0, 60.0, 70.0, 100.0])
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_ratchet_exceeds_current(self) -> None:
        # current=10, history=[200,...], ratchet=0.85*200=170 → bill=170
        result = ratchet_demand([200.0, 180.0, 190.0, 10.0])
        assert result == pytest.approx(200.0 * 0.85, rel=1e-4)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            ratchet_demand([])

    def test_single_element_no_history(self) -> None:
        assert ratchet_demand([50.0]) == pytest.approx(50.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            ratchet_demand([-10.0, 20.0])


class TestAnalysePeakDemand:
    def test_returns_dataclass(self) -> None:
        result = analyse_peak_demand([10.0, 20.0, 30.0])
        assert isinstance(result, PeakDemandResult)

    def test_peak_kw_matches(self) -> None:
        readings = [5.0, 25.0, 15.0]
        result = analyse_peak_demand(readings)
        assert result.peak_kw == 25.0

    def test_peak_index_correct(self) -> None:
        readings = [5.0, 25.0, 15.0]
        result = analyse_peak_demand(readings)
        assert result.peak_index == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_peak_demand([])

    def test_demand_charge_computed(self) -> None:
        readings = [100.0, 80.0, 90.0]
        result = analyse_peak_demand(readings, rate_per_kw=10.0)
        assert result.demand_charge == pytest.approx(1000.0)


class TestDemandResponseSavings:
    def test_full_curtailment(self) -> None:
        savings = demand_response_savings(100.0, 0.0, rate_per_kw=12.0)
        assert savings == pytest.approx(1200.0)

    def test_no_curtailment(self) -> None:
        savings = demand_response_savings(100.0, 100.0, rate_per_kw=12.0)
        assert savings == pytest.approx(0.0)

    def test_curtailed_exceeds_original_raises(self) -> None:
        with pytest.raises(ValueError):
            demand_response_savings(50.0, 100.0)

    def test_negative_original_raises(self) -> None:
        with pytest.raises(ValueError):
            demand_response_savings(-10.0, 0.0)

    @pytest.mark.parametrize(
        "orig,curtailed,rate,expected",
        [(100.0, 80.0, 10.0, 200.0), (50.0, 30.0, 15.0, 300.0)],
    )
    def test_parametrized(
        self, orig: float, curtailed: float, rate: float, expected: float
    ) -> None:
        assert demand_response_savings(orig, curtailed, rate) == pytest.approx(expected)
