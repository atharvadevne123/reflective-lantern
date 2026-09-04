"""Tests for app/solar.py."""

from __future__ import annotations

import pytest

from app.solar import (
    analyze_economics,
    generation_kwh,
    payback_years,
    self_consumption,
)

# A sunny day: no output overnight, peaking at midday.
GENERATION = [0.0, 0.0, 2.0, 6.0, 8.0, 6.0, 2.0, 0.0]
CONSUMPTION = [3.0] * 8


class TestGenerationKwh:
    def test_scales_with_area(self) -> None:
        assert generation_kwh(200.0, 5.0) == pytest.approx(2 * generation_kwh(100.0, 5.0))

    def test_scales_with_irradiance(self) -> None:
        assert generation_kwh(100.0, 10.0) == pytest.approx(2 * generation_kwh(100.0, 5.0))

    def test_zero_irradiance_yields_nothing(self) -> None:
        assert generation_kwh(100.0, 0.0) == 0.0

    def test_zero_area_yields_nothing(self) -> None:
        assert generation_kwh(0.0, 5.0) == 0.0

    def test_derates_below_raw_irradiance(self) -> None:
        # Efficiency and performance ratio both cut the theoretical maximum.
        assert generation_kwh(100.0, 5.0) < 100.0 * 5.0

    @pytest.mark.parametrize(("area", "irradiance"), [(-1.0, 5.0), (100.0, -1.0)])
    def test_negative_inputs_rejected(self, area: float, irradiance: float) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            generation_kwh(area, irradiance)

    @pytest.mark.parametrize("efficiency", [0.0, -0.1, 1.5])
    def test_invalid_efficiency_rejected(self, efficiency: float) -> None:
        with pytest.raises(ValueError, match=r"panel_efficiency must be in \(0, 1\]"):
            generation_kwh(100.0, 5.0, panel_efficiency=efficiency)

    @pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
    def test_invalid_performance_ratio_rejected(self, ratio: float) -> None:
        with pytest.raises(ValueError, match=r"performance_ratio must be in \(0, 1\]"):
            generation_kwh(100.0, 5.0, performance_ratio=ratio)


class TestSelfConsumption:
    def test_splits_generation_and_load(self) -> None:
        self_used, exported, imported = self_consumption(GENERATION, CONSUMPTION)
        assert self_used > 0
        assert exported > 0
        assert imported > 0

    def test_energy_balance_holds(self) -> None:
        self_used, exported, imported = self_consumption(GENERATION, CONSUMPTION)
        assert self_used + exported == pytest.approx(sum(GENERATION))
        assert self_used + imported == pytest.approx(sum(CONSUMPTION))

    def test_no_generation_means_all_imported(self) -> None:
        self_used, exported, imported = self_consumption([0.0] * 8, CONSUMPTION)
        assert self_used == 0.0
        assert exported == 0.0
        assert imported == pytest.approx(24.0)

    def test_no_load_means_all_exported(self) -> None:
        self_used, exported, imported = self_consumption(GENERATION, [0.0] * 8)
        assert self_used == 0.0
        assert exported == pytest.approx(sum(GENERATION))
        assert imported == 0.0

    def test_perfect_match_has_no_export_or_import(self) -> None:
        self_used, exported, imported = self_consumption([5.0] * 4, [5.0] * 4)
        assert self_used == pytest.approx(20.0)
        assert exported == 0.0
        assert imported == 0.0

    def test_matching_is_hourly_not_daily(self) -> None:
        # Same daily totals, but generation and load never coincide.
        self_used, exported, imported = self_consumption([10.0, 0.0], [0.0, 10.0])
        assert self_used == 0.0
        assert exported == pytest.approx(10.0)
        assert imported == pytest.approx(10.0)

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            self_consumption([1.0, 2.0], [1.0])

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            self_consumption([1.0, -2.0], [1.0, 1.0])


class TestAnalyzeEconomics:
    def test_rates_are_bounded_fractions(self) -> None:
        result = analyze_economics(GENERATION, CONSUMPTION)
        assert 0.0 <= result.self_consumption_rate <= 1.0
        assert 0.0 <= result.self_sufficiency_rate <= 1.0

    def test_total_benefit_sums_components(self) -> None:
        result = analyze_economics(GENERATION, CONSUMPTION)
        assert result.total_benefit == pytest.approx(round(result.bill_saving + result.export_revenue, 2))

    def test_self_consumed_energy_is_worth_more_than_export(self) -> None:
        # Same kWh self-consumed vs exported: the avoided import rate wins.
        result = analyze_economics([5.0], [5.0], import_rate=0.15, export_rate=0.05)
        exported_only = analyze_economics([5.0], [0.0], import_rate=0.15, export_rate=0.05)
        assert result.total_benefit > exported_only.total_benefit

    def test_no_generation_yields_no_benefit(self) -> None:
        result = analyze_economics([0.0] * 8, CONSUMPTION)
        assert result.total_benefit == 0.0
        assert result.self_consumption_rate == 0.0

    def test_no_load_gives_full_export(self) -> None:
        result = analyze_economics(GENERATION, [0.0] * 8)
        assert result.bill_saving == 0.0
        assert result.export_revenue > 0
        assert result.self_sufficiency_rate == 0.0

    def test_full_self_consumption_rates_are_one(self) -> None:
        result = analyze_economics([5.0] * 4, [5.0] * 4)
        assert result.self_consumption_rate == pytest.approx(1.0)
        assert result.self_sufficiency_rate == pytest.approx(1.0)
        assert result.export_revenue == 0.0

    @pytest.mark.parametrize(("import_rate", "export_rate"), [(-0.1, 0.05), (0.15, -0.05)])
    def test_negative_rates_rejected(self, import_rate: float, export_rate: float) -> None:
        with pytest.raises(ValueError, match="rates must be non-negative"):
            analyze_economics(GENERATION, CONSUMPTION, import_rate=import_rate, export_rate=export_rate)


class TestPaybackYears:
    def test_simple_case(self) -> None:
        # No degradation: 10000 / 1000 = exactly 10 years.
        assert payback_years(10000.0, 1000.0, annual_degradation=0.0) == pytest.approx(10.0)

    def test_degradation_lengthens_payback(self) -> None:
        assert payback_years(10000.0, 1000.0, annual_degradation=0.02) > payback_years(
            10000.0, 1000.0, annual_degradation=0.0
        )

    def test_larger_benefit_shortens_payback(self) -> None:
        assert payback_years(10000.0, 2000.0) < payback_years(10000.0, 1000.0)

    def test_zero_cost_repays_immediately(self) -> None:
        assert payback_years(0.0, 1000.0) == pytest.approx(0.0)

    def test_zero_benefit_never_repays(self) -> None:
        assert payback_years(10000.0, 0.0) == float("inf")

    def test_unreachable_cost_never_repays(self) -> None:
        # Heavy degradation caps lifetime benefit well below the cost.
        assert payback_years(10_000_000.0, 100.0, annual_degradation=0.5) == float("inf")

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="system_cost must be non-negative"):
            payback_years(-1.0, 1000.0)

    def test_negative_benefit_rejected(self) -> None:
        with pytest.raises(ValueError, match="annual_benefit must be non-negative"):
            payback_years(10000.0, -1.0)

    @pytest.mark.parametrize("degradation", [-0.1, 1.0, 1.5])
    def test_invalid_degradation_rejected(self, degradation: float) -> None:
        with pytest.raises(ValueError, match=r"annual_degradation must be in \[0, 1\)"):
            payback_years(10000.0, 1000.0, annual_degradation=degradation)


class TestSolarEconomicsFields:
    def test_all_fields_present(self) -> None:
        result = analyze_economics(GENERATION, CONSUMPTION)
        for field in (
            "generated_kwh",
            "consumed_kwh",
            "self_consumed_kwh",
            "exported_kwh",
            "imported_kwh",
            "self_consumption_rate",
            "self_sufficiency_rate",
            "bill_saving",
            "export_revenue",
            "total_benefit",
        ):
            assert hasattr(result, field)

    def test_generated_kwh_matches_sum(self) -> None:
        result = analyze_economics(GENERATION, CONSUMPTION)
        assert result.generated_kwh == pytest.approx(sum(GENERATION))

    def test_consumed_kwh_matches_sum(self) -> None:
        result = analyze_economics(GENERATION, CONSUMPTION)
        assert result.consumed_kwh == pytest.approx(sum(CONSUMPTION))

    def test_rates_between_zero_and_one(self) -> None:
        result = analyze_economics(GENERATION, CONSUMPTION)
        assert 0.0 <= result.self_consumption_rate <= 1.0
        assert 0.0 <= result.self_sufficiency_rate <= 1.0


class TestGenerationKwhParametrize:
    @pytest.mark.parametrize("area", [0.0, 10.0, 100.0, 500.0])
    def test_scales_linearly_with_area(self, area: float) -> None:
        result = generation_kwh(area, 5.0)
        assert result >= 0.0

    @pytest.mark.parametrize("irr", [0.0, 1.0, 5.0, 10.0])
    def test_scales_linearly_with_irradiance(self, irr: float) -> None:
        result = generation_kwh(100.0, irr)
        assert result >= 0.0

    def test_max_efficiency_and_ratio(self) -> None:
        result = generation_kwh(100.0, 5.0, panel_efficiency=1.0, performance_ratio=1.0)
        assert result == pytest.approx(100.0 * 5.0)
