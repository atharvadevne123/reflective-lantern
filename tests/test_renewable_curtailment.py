"""Tests for app.renewable_curtailment module."""

from __future__ import annotations

import pytest

from app.renewable_curtailment import (
    CurtailmentSummary,
    analyse_curtailment,
    curtailed_energy,
    curtailment_rate,
    revenue_lost,
)


class TestCurtailedEnergy:
    def test_no_curtailment(self) -> None:
        assert curtailed_energy(100.0, 100.0, 0.5) == 0.0

    def test_full_curtailment(self) -> None:
        assert curtailed_energy(100.0, 0.0, 1.0) == pytest.approx(100.0)

    def test_partial_curtailment(self) -> None:
        assert curtailed_energy(100.0, 60.0, 0.5) == pytest.approx(20.0)

    def test_negative_available_raises(self) -> None:
        with pytest.raises(ValueError):
            curtailed_energy(-10.0, 0.0, 1.0)

    def test_negative_dispatched_raises(self) -> None:
        with pytest.raises(ValueError):
            curtailed_energy(100.0, -10.0, 1.0)

    def test_zero_duration_raises(self) -> None:
        with pytest.raises(ValueError):
            curtailed_energy(100.0, 60.0, 0.0)

    @pytest.mark.parametrize(
        "avail,dispatch,dur,expected",
        [
            (200.0, 100.0, 1.0, 100.0),
            (50.0, 50.0, 2.0, 0.0),
            (100.0, 0.0, 0.25, 25.0),
        ],
    )
    def test_parametrized(self, avail: float, dispatch: float, dur: float, expected: float) -> None:
        assert curtailed_energy(avail, dispatch, dur) == pytest.approx(expected)


class TestCurtailmentRate:
    def test_zero_available_returns_zero(self) -> None:
        assert curtailment_rate(0.0, 0.0) == 0.0

    def test_full_curtailment(self) -> None:
        assert curtailment_rate(100.0, 100.0) == pytest.approx(100.0)

    def test_no_curtailment(self) -> None:
        assert curtailment_rate(100.0, 0.0) == pytest.approx(0.0)

    def test_negative_available_raises(self) -> None:
        with pytest.raises(ValueError):
            curtailment_rate(-100.0, 0.0)

    def test_curtailed_above_available_raises(self) -> None:
        with pytest.raises(ValueError):
            curtailment_rate(100.0, 110.0)

    @pytest.mark.parametrize(
        "avail,curtailed,expected_pct",
        [
            (1000.0, 100.0, 10.0),
            (500.0, 250.0, 50.0),
        ],
    )
    def test_parametrized(self, avail: float, curtailed: float, expected_pct: float) -> None:
        assert curtailment_rate(avail, curtailed) == pytest.approx(expected_pct)


class TestRevenueLost:
    def test_basic(self) -> None:
        assert revenue_lost(100.0, 0.05) == pytest.approx(5.0)

    def test_zero_curtailment(self) -> None:
        assert revenue_lost(0.0, 0.05) == 0.0

    def test_negative_curtailment_raises(self) -> None:
        with pytest.raises(ValueError):
            revenue_lost(-10.0, 0.05)

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValueError):
            revenue_lost(100.0, -0.05)


class TestAnalyseCurtailment:
    def _make_series(self) -> tuple[list[float], list[float]]:
        available = [100.0, 150.0, 200.0, 80.0, 60.0, 120.0]
        dispatched = [100.0, 100.0, 100.0, 80.0, 60.0, 100.0]
        return available, dispatched

    def test_returns_dataclass(self) -> None:
        avail, disp = self._make_series()
        result = analyse_curtailment(avail, disp)
        assert isinstance(result, CurtailmentSummary)

    def test_curtailed_positive(self) -> None:
        avail, disp = self._make_series()
        result = analyse_curtailment(avail, disp)
        assert result.total_curtailed_kwh >= 0

    def test_generated_plus_curtailed_equals_available(self) -> None:
        avail, disp = self._make_series()
        result = analyse_curtailment(avail, disp)
        assert result.total_generated_kwh + result.total_curtailed_kwh == pytest.approx(
            result.total_available_kwh, rel=1e-4
        )

    def test_mismatched_series_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_curtailment([100.0, 200.0], [100.0])

    def test_empty_series_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_curtailment([], [])

    def test_zero_interval_hours_raises(self) -> None:
        avail, disp = self._make_series()
        with pytest.raises(ValueError):
            analyse_curtailment(avail, disp, interval_hours=0.0)

    def test_no_curtailment_zero_events(self) -> None:
        avail = [100.0, 100.0, 100.0]
        disp = [100.0, 100.0, 100.0]
        result = analyse_curtailment(avail, disp)
        assert result.num_events == 0
        assert result.total_curtailed_kwh == 0.0

    def test_co2_impact_positive_when_curtailment(self) -> None:
        avail, disp = self._make_series()
        result = analyse_curtailment(avail, disp)
        if result.total_curtailed_kwh > 0:
            assert result.co2_impact_kg > 0

    def test_curtailment_rate_between_0_and_100(self) -> None:
        avail, disp = self._make_series()
        result = analyse_curtailment(avail, disp)
        assert 0 <= result.curtailment_rate_pct <= 100
