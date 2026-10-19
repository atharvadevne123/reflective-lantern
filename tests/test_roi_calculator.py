"""Tests for app.roi_calculator module."""

from __future__ import annotations

import math

import pytest

from app.roi_calculator import (
    ROIResult,
    analyse_investment,
    energy_cost_savings,
    lifetime_savings,
    net_present_value,
    roi_percent,
    simple_payback,
)


class TestSimplePayback:
    def test_basic(self) -> None:
        assert simple_payback(10000.0, 2000.0) == pytest.approx(5.0)

    def test_zero_savings_returns_inf(self) -> None:
        assert math.isinf(simple_payback(10000.0, 0.0))

    def test_negative_capex_raises(self) -> None:
        with pytest.raises(ValueError):
            simple_payback(-1000.0, 500.0)

    def test_negative_savings_raises(self) -> None:
        with pytest.raises(ValueError):
            simple_payback(1000.0, -100.0)

    @pytest.mark.parametrize(
        "capex,annual,expected",
        [(5000.0, 1000.0, 5.0), (0.0, 500.0, 0.0), (1000.0, 250.0, 4.0)],
    )
    def test_parametrized(
        self, capex: float, annual: float, expected: float
    ) -> None:
        assert simple_payback(capex, annual) == pytest.approx(expected)


class TestNetPresentValue:
    def test_zero_cashflows_negative_capex(self) -> None:
        assert net_present_value(1000.0, [], 0.08) == pytest.approx(-1000.0)

    def test_profitable_project_positive_npv(self) -> None:
        cashflows = [5000.0] * 10
        npv = net_present_value(10000.0, cashflows, 0.08)
        assert npv > 0

    def test_unprofitable_project_negative_npv(self) -> None:
        cashflows = [100.0] * 5
        npv = net_present_value(10000.0, cashflows, 0.08)
        assert npv < 0

    def test_negative_capex_raises(self) -> None:
        with pytest.raises(ValueError):
            net_present_value(-1000.0, [500.0], 0.08)

    def test_invalid_discount_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            net_present_value(1000.0, [500.0], -1.0)

    @pytest.mark.parametrize("rate", [0.05, 0.10, 0.15])
    def test_higher_discount_lower_npv(self, rate: float) -> None:
        cashflows = [1000.0] * 10
        low = net_present_value(5000.0, cashflows, 0.05)
        high = net_present_value(5000.0, cashflows, rate)
        if rate > 0.05:
            assert high <= low


class TestLifetimeSavings:
    def test_no_escalation(self) -> None:
        total = lifetime_savings(1000.0, 10, escalation_rate=0.0)
        assert total == pytest.approx(10000.0)

    def test_escalation_increases_total(self) -> None:
        flat = lifetime_savings(1000.0, 10, escalation_rate=0.0)
        escalated = lifetime_savings(1000.0, 10, escalation_rate=0.03)
        assert escalated > flat

    def test_zero_years_raises(self) -> None:
        with pytest.raises(ValueError):
            lifetime_savings(1000.0, 0)

    def test_negative_savings_raises(self) -> None:
        with pytest.raises(ValueError):
            lifetime_savings(-100.0, 10)

    @pytest.mark.parametrize("years", [1, 5, 20])
    def test_single_year_equals_annual(self, years: int) -> None:
        annual = 1000.0
        result = lifetime_savings(annual, years, escalation_rate=0.0)
        assert result == pytest.approx(annual * years)


class TestROIPercent:
    def test_breakeven_zero_roi(self) -> None:
        assert roi_percent(1000.0, 1000.0) == pytest.approx(0.0)

    def test_doubling_investment(self) -> None:
        assert roi_percent(1000.0, 2000.0) == pytest.approx(100.0)

    def test_negative_roi(self) -> None:
        assert roi_percent(1000.0, 500.0) == pytest.approx(-50.0)

    def test_zero_capex_zero_roi(self) -> None:
        assert roi_percent(0.0, 500.0) == 0.0

    def test_negative_capex_raises(self) -> None:
        with pytest.raises(ValueError):
            roi_percent(-500.0, 1000.0)

    @pytest.mark.parametrize(
        "capex,savings,expected",
        [(500.0, 1000.0, 100.0), (2000.0, 1000.0, -50.0)],
    )
    def test_parametrized(
        self, capex: float, savings: float, expected: float
    ) -> None:
        assert roi_percent(capex, savings) == pytest.approx(expected)


class TestAnalyseInvestment:
    def test_returns_roi_result(self) -> None:
        result = analyse_investment(10000.0, 2000.0)
        assert isinstance(result, ROIResult)

    def test_payback_matches(self) -> None:
        result = analyse_investment(10000.0, 2000.0, escalation_rate=0.0)
        assert result.simple_payback_years == pytest.approx(5.0)

    def test_positive_npv_for_good_project(self) -> None:
        result = analyse_investment(10000.0, 5000.0, lifetime_years=10)
        assert result.npv > 0

    def test_lifetime_savings_positive(self) -> None:
        result = analyse_investment(1000.0, 200.0)
        assert result.lifetime_savings > 0


class TestEnergyCostSavings:
    def test_basic(self) -> None:
        assert energy_cost_savings(1000.0, 0.30) == pytest.approx(300.0)

    def test_with_om_savings(self) -> None:
        assert energy_cost_savings(1000.0, 0.30, 50.0) == pytest.approx(350.0)

    def test_negative_kwh_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_cost_savings(-100.0, 0.30)

    @pytest.mark.parametrize(
        "kwh,rate,expected",
        [(500.0, 0.20, 100.0), (2000.0, 0.30, 600.0)],
    )
    def test_parametrized(self, kwh: float, rate: float, expected: float) -> None:
        assert energy_cost_savings(kwh, rate) == pytest.approx(expected)
