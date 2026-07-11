"""Energy reporting tests."""

from __future__ import annotations

import pytest

from app.reporting import estimate_savings, peak_demand_report


def test_savings_positive() -> None:
    result = estimate_savings([8.0] * 10, [10.0] * 10)
    assert result["total_saved_kwh"] == pytest.approx(20.0)
    assert result["savings_pct"] == pytest.approx(20.0)


def test_savings_negative() -> None:
    result = estimate_savings([12.0] * 5, [10.0] * 5)
    assert result["total_saved_kwh"] < 0


def test_peak_demand() -> None:
    hourly = [5.0] * 24
    hourly[14] = 25.0
    r = peak_demand_report(hourly)
    assert r["peak_hour"] == 14
    assert r["peak_kwh"] == pytest.approx(25.0)


def test_demand_factor() -> None:
    hourly = [10.0] * 24
    hourly[0] = 30.0
    r = peak_demand_report(hourly)
    assert r["demand_factor"] > 1.0


def test_savings_zero_baseline() -> None:
    result = estimate_savings([0.0], [0.0])
    assert result["savings_pct"] == 0.0


@pytest.mark.parametrize("tariff,expected_cost", [(0.10, 10.0), (0.20, 20.0), (0.15, 15.0)])
def test_savings_cost_with_tariff(tariff: float, expected_cost: float) -> None:
    result = estimate_savings([0.0] * 10, [10.0] * 10, tariff_per_kwh=tariff)
    assert result["total_saved_cost"] == pytest.approx(expected_cost)


def test_savings_returns_all_keys() -> None:
    result = estimate_savings([5.0], [10.0])
    assert "total_saved_kwh" in result
    assert "total_saved_cost" in result
    assert "savings_pct" in result


def test_peak_demand_off_peak_mean() -> None:
    hourly = [10.0] * 24
    r = peak_demand_report(hourly)
    assert r["off_peak_mean"] == pytest.approx(10.0)
    assert r["demand_factor"] == pytest.approx(1.0)


def test_estimate_savings_single_value() -> None:
    result = estimate_savings([5.0], [8.0])
    assert result["total_saved_kwh"] == pytest.approx(3.0)
    assert result["savings_pct"] == pytest.approx(37.5)


def test_estimate_savings_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        estimate_savings([1.0, 2.0], [1.0])


def test_peak_demand_empty_raises() -> None:
    with pytest.raises(ValueError):
        peak_demand_report([])


@pytest.mark.parametrize("n_hours", [1, 12, 24, 48])
def test_peak_demand_various_lengths(n_hours: int) -> None:
    hourly = list(range(n_hours, 0, -1))
    r = peak_demand_report([float(v) for v in hourly])
    assert r["peak_hour"] == 0  # largest is first element
