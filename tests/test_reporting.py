"""Energy reporting tests."""
from __future__ import annotations

import pytest

from app.reporting import estimate_savings, peak_demand_report


def test_savings_positive():
    result = estimate_savings([8.0]*10, [10.0]*10)
    assert result["total_saved_kwh"] == pytest.approx(20.0)
    assert result["savings_pct"] == pytest.approx(20.0)

def test_savings_negative():
    result = estimate_savings([12.0]*5, [10.0]*5)
    assert result["total_saved_kwh"] < 0

def test_peak_demand():
    hourly = [5.0]*24
    hourly[14] = 25.0
    r = peak_demand_report(hourly)
    assert r["peak_hour"] == 14
    assert r["peak_kwh"] == pytest.approx(25.0)

def test_demand_factor():
    hourly = [10.0]*24
    hourly[0] = 30.0
    r = peak_demand_report(hourly)
    assert r["demand_factor"] > 1.0

def test_savings_zero_baseline():
    result = estimate_savings([0.0], [0.0])
    assert result["savings_pct"] == 0.0
