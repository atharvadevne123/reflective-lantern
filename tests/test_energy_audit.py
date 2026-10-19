"""Tests for app.energy_audit module."""

from __future__ import annotations

import pytest

from app.energy_audit import (
    AuditLevel,
    AuditOpportunity,
    AuditReport,
    EndUseCategory,
    build_end_use_breakdown,
    end_use_percentage,
    energy_use_intensity,
    opportunity_payback,
    prioritise_opportunities,
    run_audit,
)


def _opportunity(payback: float, cost: float = 1000.0, savings: float = 100.0) -> AuditOpportunity:
    return AuditOpportunity(
        description="test",
        category=EndUseCategory.LIGHTING,
        estimated_annual_savings_kwh=500.0,
        estimated_annual_savings_cost=savings,
        implementation_cost=cost,
        simple_payback_years=payback,
        priority="high",
    )


class TestEnergyUseIntensity:
    def test_basic(self) -> None:
        assert energy_use_intensity(10000.0, 100.0) == pytest.approx(100.0)

    def test_negative_kwh_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_use_intensity(-100.0, 100.0)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_use_intensity(1000.0, 0.0)

    @pytest.mark.parametrize("area", [50.0, 200.0, 1000.0])
    def test_positive_result(self, area: float) -> None:
        assert energy_use_intensity(50000.0, area) > 0.0


class TestEndUsePercentage:
    def test_half(self) -> None:
        assert end_use_percentage(500.0, 1000.0) == pytest.approx(50.0)

    def test_zero_end_use(self) -> None:
        assert end_use_percentage(0.0, 1000.0) == pytest.approx(0.0)

    def test_zero_total_raises(self) -> None:
        with pytest.raises(ValueError):
            end_use_percentage(100.0, 0.0)

    def test_negative_end_use_raises(self) -> None:
        with pytest.raises(ValueError):
            end_use_percentage(-100.0, 1000.0)

    @pytest.mark.parametrize("eu,total", [(100.0, 1000.0), (250.0, 500.0)])
    def test_within_0_100(self, eu: float, total: float) -> None:
        assert 0.0 <= end_use_percentage(eu, total) <= 100.0


class TestOpportunityPayback:
    def test_basic(self) -> None:
        assert opportunity_payback(500.0, 2500.0) == pytest.approx(5.0)

    def test_zero_savings_raises(self) -> None:
        with pytest.raises(ValueError):
            opportunity_payback(0.0, 1000.0)

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError):
            opportunity_payback(500.0, -100.0)

    def test_zero_cost_zero_payback(self) -> None:
        assert opportunity_payback(500.0, 0.0) == pytest.approx(0.0)


class TestPrioritiseOpportunities:
    def test_sorted_by_payback(self) -> None:
        ops = [_opportunity(5.0), _opportunity(2.0), _opportunity(8.0)]
        result = prioritise_opportunities(ops)
        paybacks = [o.simple_payback_years for o in result]
        assert paybacks == sorted(paybacks)

    def test_empty_returns_empty(self) -> None:
        assert prioritise_opportunities([]) == []

    def test_single_unchanged(self) -> None:
        ops = [_opportunity(3.0)]
        assert prioritise_opportunities(ops)[0].simple_payback_years == pytest.approx(3.0)


class TestBuildEndUseBreakdown:
    def test_returns_list(self) -> None:
        end_uses = {EndUseCategory.HVAC: 5000.0, EndUseCategory.LIGHTING: 2000.0}
        result = build_end_use_breakdown(end_uses, 0.28)
        assert len(result) == 2

    def test_pct_sums_to_100(self) -> None:
        end_uses = {EndUseCategory.HVAC: 5000.0, EndUseCategory.LIGHTING: 5000.0}
        result = build_end_use_breakdown(end_uses, 0.28)
        total_pct = sum(b.pct_of_total for b in result)
        assert total_pct == pytest.approx(100.0)

    def test_negative_consumption_raises(self) -> None:
        with pytest.raises(ValueError):
            build_end_use_breakdown({EndUseCategory.HVAC: -100.0}, 0.28)

    def test_zero_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            build_end_use_breakdown({EndUseCategory.HVAC: 100.0}, 0.0)


class TestRunAudit:
    def test_returns_report(self) -> None:
        end_uses = {EndUseCategory.HVAC: 5000.0, EndUseCategory.LIGHTING: 2000.0}
        result = run_audit("B001", 200.0, end_uses, 0.28, [])
        assert isinstance(result, AuditReport)

    def test_building_id_preserved(self) -> None:
        end_uses = {EndUseCategory.HVAC: 5000.0}
        result = run_audit("B99", 100.0, end_uses, 0.28, [])
        assert result.building_id == "B99"

    def test_opportunities_sorted(self) -> None:
        end_uses = {EndUseCategory.HVAC: 5000.0}
        ops = [_opportunity(5.0), _opportunity(2.0)]
        result = run_audit("B1", 100.0, end_uses, 0.28, ops)
        paybacks = [o.simple_payback_years for o in result.opportunities]
        assert paybacks == sorted(paybacks)

    @pytest.mark.parametrize("area", [100.0, 500.0, 2000.0])
    def test_eui_positive(self, area: float) -> None:
        end_uses = {EndUseCategory.HVAC: 10000.0}
        result = run_audit("B", area, end_uses, 0.28, [])
        assert result.energy_use_intensity_kwh_per_m2 > 0.0
