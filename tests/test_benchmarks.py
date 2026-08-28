"""Tests for app/benchmarks.py."""

from __future__ import annotations

import pytest

from app.benchmarks import eui_percentile_category, target_eui


def test_compute_eui_basic() -> None:
    from app.benchmarks import compute_eui

    result = compute_eui(10000.0, 100.0)
    assert abs(result - 100.0) < 0.01


def test_compute_eui_zero_area_raises() -> None:
    from app.benchmarks import compute_eui

    with pytest.raises(ValueError, match="floor_area_sqm"):
        compute_eui(1000.0, 0.0)


def test_compute_eui_negative_area_raises() -> None:
    from app.benchmarks import compute_eui

    with pytest.raises(ValueError, match="floor_area_sqm"):
        compute_eui(1000.0, -50.0)


def test_compute_eui_negative_kwh_raises() -> None:
    from app.benchmarks import compute_eui

    with pytest.raises(ValueError, match="annual_kwh"):
        compute_eui(-100.0, 50.0)


def test_compute_eui_zero_kwh() -> None:
    from app.benchmarks import compute_eui

    assert compute_eui(0.0, 100.0) == 0.0


@pytest.mark.parametrize(
    "annual_kwh,sqm,expected",
    [
        (17600.0, 100.0, 176.0),
        (5000.0, 50.0, 100.0),
        (25000.0, 125.0, 200.0),
    ],
)
def test_compute_eui_parametrized(annual_kwh, sqm, expected) -> None:
    from app.benchmarks import compute_eui

    assert abs(compute_eui(annual_kwh, sqm) - expected) < 0.01


def test_benchmark_eui_excellent() -> None:
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(100.0, "office")  # 100 vs 176 benchmark -> ratio ~0.57
    assert result["rating"] == "excellent"


def test_benchmark_eui_poor() -> None:
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(300.0, "office")  # 300 vs 176 benchmark -> ratio ~1.7
    assert result["rating"] == "poor"


def test_benchmark_eui_average() -> None:
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(176.0, "office")  # ratio = 1.0
    assert result["rating"] == "average"


def test_benchmark_eui_unknown_type_uses_default() -> None:
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(200.0, "unknown_type")
    assert result["benchmark_eui"] == 200.0  # default benchmark


def test_benchmark_eui_has_all_keys() -> None:
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(150.0, "office")
    assert "building_type" in result
    assert "eui" in result
    assert "benchmark_eui" in result
    assert "ratio" in result
    assert "rating" in result


def test_annual_to_monthly_estimate_sums_to_annual() -> None:
    from app.benchmarks import annual_to_monthly_estimate

    result = annual_to_monthly_estimate(12000.0)
    assert len(result) == 12
    assert abs(sum(result) - 12000.0) < 1.0


def test_annual_to_monthly_estimate_custom_profile() -> None:
    from app.benchmarks import annual_to_monthly_estimate

    profile = [1.0] * 12
    result = annual_to_monthly_estimate(1200.0, profile=profile)
    assert all(abs(v - 100.0) < 0.1 for v in result)


def test_annual_to_monthly_estimate_invalid_profile() -> None:
    from app.benchmarks import annual_to_monthly_estimate

    with pytest.raises(ValueError, match="12 elements"):
        annual_to_monthly_estimate(1000.0, profile=[1.0] * 11)


def test_list_building_types() -> None:
    from app.benchmarks import list_building_types

    types = list_building_types()
    assert "office" in types
    assert "hospital" in types
    assert types == sorted(types)


@pytest.mark.parametrize("building_type", ["office", "hotel", "school", "warehouse"])
def test_benchmark_eui_known_types(building_type) -> None:
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS, benchmark_eui

    result = benchmark_eui(200.0, building_type)
    assert result["benchmark_eui"] == ASHRAE_EUI_BENCHMARKS[building_type]
    assert result["rating"] in ("excellent", "good", "average", "poor")


def test_site_eui_basic() -> None:
    from app.benchmarks import site_eui

    result = site_eui(10000.0, 1076.39)  # ~100 sqm in sqft
    assert result == pytest.approx(100.0, rel=1e-2)


def test_site_eui_zero_area_raises() -> None:
    from app.benchmarks import site_eui

    with pytest.raises(ValueError, match="floor_area_sqft"):
        site_eui(1000.0, 0.0)


def test_site_eui_negative_kwh_raises() -> None:
    from app.benchmarks import site_eui

    with pytest.raises(ValueError, match="annual_kwh"):
        site_eui(-1.0, 500.0)


def test_compare_buildings_ranks_by_eui() -> None:
    from app.benchmarks import compare_buildings

    buildings = [
        {"name": "A", "annual_kwh": 20000.0, "floor_area_sqm": 100.0},
        {"name": "B", "annual_kwh": 10000.0, "floor_area_sqm": 100.0},
    ]
    results = compare_buildings(buildings)
    assert results[0]["name"] == "B"
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2


def test_compare_buildings_empty() -> None:
    from app.benchmarks import compare_buildings

    assert compare_buildings([]) == []


def test_compare_buildings_single_building() -> None:
    from app.benchmarks import compare_buildings

    buildings = [{"name": "X", "annual_kwh": 5000.0, "floor_area_sqm": 50.0}]
    results = compare_buildings(buildings)
    assert len(results) == 1
    assert results[0]["rank"] == 1
    assert "rating" in results[0]


def test_energy_intensity_ratio_basic() -> None:
    from app.benchmarks import energy_intensity_ratio

    assert energy_intensity_ratio(150.0, 200.0) == pytest.approx(0.75)


def test_energy_intensity_ratio_equal() -> None:
    from app.benchmarks import energy_intensity_ratio

    assert energy_intensity_ratio(200.0, 200.0) == pytest.approx(1.0)


def test_energy_intensity_ratio_zero_reference() -> None:
    from app.benchmarks import energy_intensity_ratio

    assert energy_intensity_ratio(150.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "actual,ref,expected",
    [
        (100.0, 200.0, 0.5),
        (200.0, 100.0, 2.0),
        (0.0, 100.0, 0.0),
    ],
)
def test_energy_intensity_ratio_parametrized(actual: float, ref: float, expected: float) -> None:
    from app.benchmarks import energy_intensity_ratio

    assert energy_intensity_ratio(actual, ref) == pytest.approx(expected, rel=1e-4)


class TestEfficiencyGap:
    def test_over_target(self) -> None:
        # actual 120, target 100 → 20% over
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(120.0, 100.0) == pytest.approx(20.0, rel=1e-4)

    def test_under_target(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(80.0, 100.0) == pytest.approx(-20.0, rel=1e-4)

    def test_equal_values(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(100.0, 100.0) == pytest.approx(0.0, abs=1e-9)

    def test_zero_target_returns_zero(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(50.0, 0.0) == 0.0

    def test_zero_actual(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(0.0, 100.0) == pytest.approx(-100.0, rel=1e-4)

    def test_large_gap(self) -> None:
        from app.benchmarks import efficiency_gap

        result = efficiency_gap(500.0, 200.0)
        assert result == pytest.approx(150.0, rel=1e-4)


def test_list_building_types_not_empty() -> None:
    from app.benchmarks import list_building_types

    types = list_building_types()
    assert isinstance(types, list)
    assert len(types) > 0


def test_list_building_types_are_strings() -> None:
    from app.benchmarks import list_building_types

    for t in list_building_types():
        assert isinstance(t, str)


def test_site_eui_basic_inline() -> None:
    from app.benchmarks import site_eui

    result = site_eui(10000.0, 1000.0)
    assert isinstance(result, float)
    assert result > 0


def test_annual_to_monthly_estimate_length() -> None:
    from app.benchmarks import annual_to_monthly_estimate

    result = annual_to_monthly_estimate(12000.0)
    assert len(result) == 12


def test_annual_to_monthly_estimate_sums_to_annual_approx() -> None:
    from app.benchmarks import annual_to_monthly_estimate

    annual = 12000.0
    result = annual_to_monthly_estimate(annual)
    assert sum(result) == pytest.approx(annual, rel=1e-3)


@pytest.mark.parametrize("building_type", ["hospital", "hotel", "default"])
def test_benchmark_eui_known_type(building_type: str) -> None:
    from app.benchmarks import benchmark_eui

    result = benchmark_eui(100.0, building_type)
    assert isinstance(result, dict)
    assert "benchmark_eui" in result
    assert result["benchmark_eui"] > 0


class TestEnergyIntensityRatio:
    def test_below_reference(self) -> None:
        from app.benchmarks import energy_intensity_ratio

        assert energy_intensity_ratio(80.0, 100.0) == pytest.approx(0.8, rel=1e-4)

    def test_above_reference(self) -> None:
        from app.benchmarks import energy_intensity_ratio

        assert energy_intensity_ratio(120.0, 100.0) == pytest.approx(1.2, rel=1e-4)

    def test_zero_reference_returns_zero(self) -> None:
        from app.benchmarks import energy_intensity_ratio

        assert energy_intensity_ratio(100.0, 0.0) == 0.0

    def test_equal_returns_one(self) -> None:
        from app.benchmarks import energy_intensity_ratio

        assert energy_intensity_ratio(100.0, 100.0) == pytest.approx(1.0)


class TestEfficiencyGapNew:
    def test_positive_gap(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(120.0, 100.0) == pytest.approx(20.0)

    def test_negative_gap(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(80.0, 100.0) == pytest.approx(-20.0)

    def test_zero_target(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(100.0, 0.0) == 0.0

    def test_no_gap(self) -> None:
        from app.benchmarks import efficiency_gap

        assert efficiency_gap(100.0, 100.0) == pytest.approx(0.0)


class TestCarbonIntensityBenchmark:
    def test_basic(self) -> None:
        from app.benchmarks import carbon_intensity_benchmark

        result = carbon_intensity_benchmark(1000.0, 0.5, 100.0)
        assert result == pytest.approx(5.0)

    def test_non_positive_kwh(self) -> None:
        from app.benchmarks import carbon_intensity_benchmark

        with pytest.raises(ValueError):
            carbon_intensity_benchmark(0.0, 0.5, 100.0)

    def test_non_positive_area(self) -> None:
        from app.benchmarks import carbon_intensity_benchmark

        with pytest.raises(ValueError):
            carbon_intensity_benchmark(1000.0, 0.5, 0.0)


class TestPercentageBelowBenchmark:
    def test_better_than_benchmark(self) -> None:
        from app.benchmarks import percentage_below_benchmark

        result = percentage_below_benchmark(80.0, 100.0)
        assert result == pytest.approx(20.0)

    def test_worse_than_benchmark(self) -> None:
        from app.benchmarks import percentage_below_benchmark

        result = percentage_below_benchmark(120.0, 100.0)
        assert result < 0

    def test_zero_benchmark_raises(self) -> None:
        from app.benchmarks import percentage_below_benchmark

        with pytest.raises(ValueError):
            percentage_below_benchmark(50.0, 0.0)


class TestWeightedAverageEui:
    def test_equal_weights(self) -> None:
        from app.benchmarks import weighted_average_eui

        result = weighted_average_eui([100.0, 200.0], [1.0, 1.0])
        assert result == pytest.approx(150.0)

    def test_empty_raises(self) -> None:
        from app.benchmarks import weighted_average_eui

        with pytest.raises(ValueError):
            weighted_average_eui([], [])

    def test_length_mismatch(self) -> None:
        from app.benchmarks import weighted_average_eui

        with pytest.raises(ValueError):
            weighted_average_eui([1.0, 2.0], [1.0])


class TestEnergyStarScore:
    def test_median_gets_50(self) -> None:
        from app.benchmarks import energy_star_score

        result = energy_star_score(100.0, 100.0, 50.0)
        assert result == pytest.approx(50.0)

    def test_best_gets_100(self) -> None:
        from app.benchmarks import energy_star_score

        result = energy_star_score(50.0, 100.0, 50.0)
        assert result == pytest.approx(100.0)

    def test_score_clamped_at_zero(self) -> None:
        from app.benchmarks import energy_star_score

        result = energy_star_score(200.0, 100.0, 50.0)
        assert result == pytest.approx(0.0)

    def test_equal_median_best_raises(self) -> None:
        from app.benchmarks import energy_star_score

        with pytest.raises(ValueError):
            energy_star_score(100.0, 100.0, 100.0)


def test_target_eui_zero_improvement() -> None:
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS

    result = target_eui("office", improvement_pct=0.0)
    assert result == pytest.approx(ASHRAE_EUI_BENCHMARKS["office"])


def test_target_eui_20pct_improvement() -> None:
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS

    result = target_eui("office", improvement_pct=20.0)
    assert result == pytest.approx(ASHRAE_EUI_BENCHMARKS["office"] * 0.8)


def test_target_eui_100pct_is_zero() -> None:
    assert target_eui("office", improvement_pct=100.0) == pytest.approx(0.0)


def test_target_eui_invalid_pct_raises() -> None:
    with pytest.raises(ValueError):
        target_eui("office", improvement_pct=101.0)


def test_target_eui_negative_pct_raises() -> None:
    with pytest.raises(ValueError):
        target_eui("office", improvement_pct=-1.0)


def test_target_eui_unknown_type_uses_default() -> None:
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS

    result = target_eui("unknown_type", improvement_pct=10.0)
    assert result == pytest.approx(ASHRAE_EUI_BENCHMARKS["default"] * 0.9)


def test_eui_percentile_top_25() -> None:
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS

    low_eui = ASHRAE_EUI_BENCHMARKS["office"] * 0.4
    assert eui_percentile_category(low_eui, "office") == "top_25"


def test_eui_percentile_median() -> None:
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS

    mid_eui = ASHRAE_EUI_BENCHMARKS["office"] * 0.7
    assert eui_percentile_category(mid_eui, "office") == "median"


def test_eui_percentile_average() -> None:
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS

    avg_eui = ASHRAE_EUI_BENCHMARKS["office"] * 1.0
    assert eui_percentile_category(avg_eui, "office") == "average"


def test_eui_percentile_bottom_25() -> None:
    from app.benchmarks import ASHRAE_EUI_BENCHMARKS

    high_eui = ASHRAE_EUI_BENCHMARKS["office"] * 1.5
    assert eui_percentile_category(high_eui, "office") == "bottom_25"


@pytest.mark.parametrize("building_type", ["office", "retail", "school", "hospital"])
def test_target_eui_known_types(building_type: str) -> None:
    result = target_eui(building_type, improvement_pct=10.0)
    assert result > 0


class TestEuiImprovementNeeded:
    def test_already_at_target_returns_zero(self) -> None:
        from app.benchmarks import ASHRAE_EUI_BENCHMARKS, eui_improvement_needed

        benchmark = ASHRAE_EUI_BENCHMARKS["office"]
        target = benchmark * 0.5
        assert eui_improvement_needed(target, "office", target_percentile=50.0) == pytest.approx(0.0)

    def test_above_target_returns_positive(self) -> None:
        from app.benchmarks import ASHRAE_EUI_BENCHMARKS, eui_improvement_needed

        benchmark = ASHRAE_EUI_BENCHMARKS["office"]
        result = eui_improvement_needed(benchmark, "office", target_percentile=50.0)
        assert result > 0.0

    def test_invalid_percentile_raises(self) -> None:
        from app.benchmarks import eui_improvement_needed

        with pytest.raises(ValueError):
            eui_improvement_needed(100.0, "office", target_percentile=0.0)

    @pytest.mark.parametrize("tp", [25.0, 50.0, 80.0, 100.0])
    def test_parametrized_percentile(self, tp) -> None:
        from app.benchmarks import eui_improvement_needed

        result = eui_improvement_needed(200.0, "office", target_percentile=tp)
        assert result >= 0.0


class TestNormaliseEui:
    def test_at_benchmark_is_one(self) -> None:
        from app.benchmarks import ASHRAE_EUI_BENCHMARKS, normalise_eui

        benchmark = ASHRAE_EUI_BENCHMARKS["office"]
        assert normalise_eui(benchmark, "office") == pytest.approx(1.0)

    def test_half_benchmark_is_half(self) -> None:
        from app.benchmarks import ASHRAE_EUI_BENCHMARKS, normalise_eui

        benchmark = ASHRAE_EUI_BENCHMARKS["office"]
        assert normalise_eui(benchmark / 2.0, "office") == pytest.approx(0.5)

    def test_unknown_type_uses_default(self) -> None:
        from app.benchmarks import normalise_eui

        result = normalise_eui(100.0, "nonexistent_type")
        assert result > 0.0


class TestEuiSavingsPotential:
    def test_returns_expected_keys(self) -> None:
        from app.benchmarks import eui_savings_potential

        result = eui_savings_potential(200.0, 1000.0)
        assert "saved_kwh_per_year" in result
        assert "saved_cost_per_year" in result
        assert "new_eui" in result

    def test_20_pct_improvement(self) -> None:
        from app.benchmarks import eui_savings_potential

        result = eui_savings_potential(100.0, 1000.0, improvement_pct=20.0)
        assert result["saved_kwh_per_year"] == pytest.approx(20_000.0)
        assert result["new_eui"] == pytest.approx(80.0)

    @pytest.mark.parametrize("pct", [10.0, 20.0, 50.0])
    def test_savings_positive_for_positive_inputs(self, pct) -> None:
        from app.benchmarks import eui_savings_potential

        result = eui_savings_potential(200.0, 500.0, improvement_pct=pct)
        assert result["saved_cost_per_year"] > 0.0


class TestEnergyUseIntensityDelta:
    def test_improvement(self) -> None:
        from app.benchmarks import energy_use_intensity_delta

        result = energy_use_intensity_delta(200.0, 150.0)
        assert result["absolute_delta"] == pytest.approx(50.0, abs=0.01)
        assert result["pct_change"] == pytest.approx(25.0, abs=0.01)

    def test_no_change(self) -> None:
        from app.benchmarks import energy_use_intensity_delta

        result = energy_use_intensity_delta(100.0, 100.0)
        assert result["absolute_delta"] == pytest.approx(0.0)
        assert result["pct_change"] == pytest.approx(0.0)

    def test_zero_baseline_returns_zero_pct(self) -> None:
        from app.benchmarks import energy_use_intensity_delta

        result = energy_use_intensity_delta(0.0, 10.0)
        assert result["pct_change"] == 0.0


class TestStarRatingFromScore:
    def test_five_stars(self) -> None:
        from app.benchmarks import star_rating_from_score

        assert star_rating_from_score(0.95) == 5

    def test_one_star(self) -> None:
        from app.benchmarks import star_rating_from_score

        assert star_rating_from_score(0.1) == 1

    def test_boundary_values(self) -> None:
        from app.benchmarks import star_rating_from_score

        assert star_rating_from_score(0.9) == 5
        assert star_rating_from_score(0.75) == 4
        assert star_rating_from_score(0.55) == 3


class TestPortfolioEuiSummary:
    def test_basic(self) -> None:
        from app.benchmarks import portfolio_eui_summary

        result = portfolio_eui_summary([{"eui": 100.0}, {"eui": 200.0}])
        assert result["mean_eui"] == pytest.approx(150.0)
        assert result["min_eui"] == pytest.approx(100.0)
        assert result["max_eui"] == pytest.approx(200.0)

    def test_empty_returns_zeros(self) -> None:
        from app.benchmarks import portfolio_eui_summary

        result = portfolio_eui_summary([])
        assert result == {"mean_eui": 0.0, "min_eui": 0.0, "max_eui": 0.0}


@pytest.mark.parametrize(
    "annual_kwh,floor_area,expected_eui",
    [
        (10000.0, 100.0, 100.0),
        (50000.0, 500.0, 100.0),
        (20000.0, 200.0, 100.0),
    ],
)
def test_compute_eui_scales_linearly(annual_kwh: float, floor_area: float, expected_eui: float) -> None:
    from app.benchmarks import compute_eui

    assert compute_eui(annual_kwh, floor_area) == pytest.approx(expected_eui, abs=0.1)


@pytest.mark.parametrize(
    "actual_eui,benchmark,expected_sign",
    [
        (80.0, 100.0, "positive"),
        (120.0, 100.0, "negative"),
        (100.0, 100.0, "zero"),
    ],
)
def test_percentage_below_benchmark_sign(actual_eui: float, benchmark: float, expected_sign: str) -> None:
    from app.benchmarks import percentage_below_benchmark

    result = percentage_below_benchmark(actual_eui, benchmark)
    if expected_sign == "positive":
        assert result > 0.0
    elif expected_sign == "negative":
        assert result < 0.0
    else:
        assert result == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("score", [0.0, 50.0, 75.0, 100.0])
def test_benchmark_score_label_returns_string(score: float) -> None:
    from app.benchmarks import benchmark_score_label

    label = benchmark_score_label(score)
    assert isinstance(label, str)
    assert len(label) > 0


@pytest.mark.parametrize(
    "euis,weights",
    [
        ([100.0, 200.0], [0.5, 0.5]),
        ([100.0, 200.0, 300.0], [1.0, 2.0, 1.0]),
    ],
)
def test_weighted_average_eui_in_range(euis: list, weights: list) -> None:
    from app.benchmarks import weighted_average_eui

    result = weighted_average_eui(euis, weights)
    assert min(euis) <= result <= max(euis)


class TestEuiImprovementRate:
    def test_improvement(self) -> None:
        from app.benchmarks import eui_improvement_rate

        assert eui_improvement_rate(100.0, 80.0) == pytest.approx(20.0)

    def test_no_change(self) -> None:
        from app.benchmarks import eui_improvement_rate

        assert eui_improvement_rate(100.0, 100.0) == pytest.approx(0.0)

    def test_regression(self) -> None:
        from app.benchmarks import eui_improvement_rate

        assert eui_improvement_rate(100.0, 120.0) == pytest.approx(-20.0)

    def test_zero_baseline_raises(self) -> None:
        from app.benchmarks import eui_improvement_rate

        with pytest.raises(ValueError):
            eui_improvement_rate(0.0, 80.0)


class TestNormalisedEui:
    def test_basic(self) -> None:
        from app.benchmarks import normalised_eui

        result = normalised_eui(10000.0, 100.0, 2000.0)
        assert result == pytest.approx(0.05)

    def test_zero_area_raises(self) -> None:
        from app.benchmarks import normalised_eui

        with pytest.raises(ValueError):
            normalised_eui(10000.0, 0.0, 2000.0)

    def test_zero_hours_raises(self) -> None:
        from app.benchmarks import normalised_eui

        with pytest.raises(ValueError):
            normalised_eui(10000.0, 100.0, 0.0)


class TestSavingsToInvestmentRatio:
    def test_basic(self) -> None:
        from app.benchmarks import savings_to_investment_ratio

        result = savings_to_investment_ratio(5000.0, 0.15, 10000.0)
        assert result == pytest.approx(0.075)

    def test_zero_savings(self) -> None:
        from app.benchmarks import savings_to_investment_ratio

        assert savings_to_investment_ratio(0.0, 0.15, 10000.0) == pytest.approx(0.0)

    def test_zero_investment_raises(self) -> None:
        from app.benchmarks import savings_to_investment_ratio

        with pytest.raises(ValueError):
            savings_to_investment_ratio(5000.0, 0.15, 0.0)
