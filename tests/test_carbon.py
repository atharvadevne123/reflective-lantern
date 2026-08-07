"""Tests for app/carbon.py."""

from __future__ import annotations

import pytest

from app.carbon import carbon_per_sqm, carbon_reduction_pct


def test_kwh_to_co2_kg_default_region() -> None:
    from app.carbon import kwh_to_co2_kg

    result = kwh_to_co2_kg(100.0)
    assert result > 0.0
    assert result == pytest.approx(35.0, rel=0.01)


def test_kwh_to_co2_kg_pacific_nw() -> None:
    from app.carbon import kwh_to_co2_kg

    result = kwh_to_co2_kg(100.0, region="pacific_nw")
    assert result == pytest.approx(10.0, rel=0.01)


def test_kwh_to_co2_kg_negative_raises() -> None:
    from app.carbon import kwh_to_co2_kg

    with pytest.raises(ValueError, match="non-negative"):
        kwh_to_co2_kg(-5.0)


def test_kwh_to_co2_kg_zero() -> None:
    from app.carbon import kwh_to_co2_kg

    assert kwh_to_co2_kg(0.0) == 0.0


def test_co2_kg_to_tonnes() -> None:
    from app.carbon import co2_kg_to_tonnes

    assert co2_kg_to_tonnes(1000.0) == pytest.approx(1.0, rel=0.001)


def test_trees_equivalent_basic() -> None:
    from app.carbon import trees_equivalent

    result = trees_equivalent(20.0)
    assert result == pytest.approx(1.0, rel=0.01)


def test_trees_equivalent_zero() -> None:
    from app.carbon import trees_equivalent

    assert trees_equivalent(0.0) == 0.0


def test_annual_carbon_report_valid() -> None:
    from app.carbon import annual_carbon_report

    monthly = [100.0] * 12
    result = annual_carbon_report(monthly)
    assert result["total_kwh"] == pytest.approx(1200.0, rel=0.01)
    assert len(result["monthly_co2_kg"]) == 12
    assert result["total_co2_kg"] > 0


def test_annual_carbon_report_wrong_months() -> None:
    from app.carbon import annual_carbon_report

    with pytest.raises(ValueError, match="12 elements"):
        annual_carbon_report([100.0] * 11)


def test_annual_carbon_report_has_trees() -> None:
    from app.carbon import annual_carbon_report

    result = annual_carbon_report([500.0] * 12)
    assert "trees_equivalent" in result
    assert result["trees_equivalent"] >= 0


def test_carbon_savings_basic() -> None:
    from app.carbon import carbon_savings

    result = carbon_savings(actual_kwh=80.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == pytest.approx(20.0, rel=0.01)
    assert result["saved_co2_kg"] > 0


def test_carbon_savings_no_savings() -> None:
    from app.carbon import carbon_savings

    result = carbon_savings(actual_kwh=100.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == 0.0
    assert result["saved_co2_kg"] == 0.0


def test_carbon_savings_actual_exceeds_baseline() -> None:
    from app.carbon import carbon_savings

    result = carbon_savings(actual_kwh=150.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == 0.0


@pytest.mark.parametrize("region", ["northeast", "midwest", "south", "west", "pacific_nw"])
def test_kwh_to_co2_kg_known_regions(region) -> None:
    from app.carbon import GRID_CARBON_INTENSITY, kwh_to_co2_kg

    result = kwh_to_co2_kg(100.0, region=region)
    expected = GRID_CARBON_INTENSITY[region] * 100.0
    assert abs(result - expected) < 0.01


@pytest.mark.parametrize(
    "region,factor",
    [
        ("northeast", 0.25),
        ("midwest", 0.45),
        ("pacific_nw", 0.10),
        ("texas", 0.40),
        ("default", 0.35),
    ],
)
def test_kwh_to_co2_kg_known_regions_parametrized(region: str, factor: float) -> None:
    from app.carbon import kwh_to_co2_kg

    assert kwh_to_co2_kg(100.0, region=region) == pytest.approx(100.0 * factor, rel=1e-4)


def test_kwh_to_co2_kg_case_insensitive() -> None:
    from app.carbon import kwh_to_co2_kg

    assert kwh_to_co2_kg(100.0, region="PACIFIC_NW") == pytest.approx(kwh_to_co2_kg(100.0, region="pacific_nw"))


def test_kwh_to_co2_kg_unknown_region_uses_default() -> None:
    from app.carbon import GRID_CARBON_INTENSITY, kwh_to_co2_kg

    result = kwh_to_co2_kg(100.0, region="atlantis")
    assert result == pytest.approx(100.0 * GRID_CARBON_INTENSITY["default"])


def test_carbon_savings_no_actual_reduction() -> None:
    from app.carbon import carbon_savings

    result = carbon_savings(actual_kwh=100.0, baseline_kwh=100.0)
    assert result["saved_kwh"] == pytest.approx(0.0)
    assert result["saved_co2_kg"] == pytest.approx(0.0)


@pytest.mark.parametrize("kwh", [0.0, 1.0, 100.0, 10000.0])
def test_kwh_to_co2_kg_non_negative_output(kwh: float) -> None:
    from app.carbon import kwh_to_co2_kg

    assert kwh_to_co2_kg(kwh) >= 0.0


@pytest.mark.parametrize(
    "kg,expected_tonnes",
    [
        (0.0, 0.0),
        (500.0, 0.5),
        (2000.0, 2.0),
    ],
)
def test_co2_kg_to_tonnes_parametrized(kg: float, expected_tonnes: float) -> None:
    from app.carbon import co2_kg_to_tonnes

    assert co2_kg_to_tonnes(kg) == pytest.approx(expected_tonnes, rel=1e-4)


def test_annual_carbon_report_returns_region() -> None:
    from app.carbon import annual_carbon_report

    result = annual_carbon_report([100.0] * 12, region="west")
    assert result["region"] == "west"


def test_annual_carbon_report_empty_list_raises() -> None:
    from app.carbon import annual_carbon_report

    with pytest.raises(ValueError):
        annual_carbon_report([])


@pytest.mark.parametrize(
    "actual,baseline,expected_saved",
    [
        (50.0, 100.0, 50.0),
        (0.0, 100.0, 100.0),
        (100.0, 50.0, 0.0),
    ],
)
def test_carbon_savings_parametrized(actual: float, baseline: float, expected_saved: float) -> None:
    from app.carbon import carbon_savings

    result = carbon_savings(actual_kwh=actual, baseline_kwh=baseline)
    assert result["saved_kwh"] == pytest.approx(expected_saved, rel=1e-4)


def test_trees_equivalent_large_value() -> None:
    from app.carbon import trees_equivalent

    result = trees_equivalent(1_000_000.0)
    assert result > 0
    assert isinstance(result, float)


def test_daily_carbon_estimate_basic() -> None:
    from app.carbon import daily_carbon_estimate

    hourly = [10.0] * 24
    result = daily_carbon_estimate(hourly)
    assert result["total_kwh"] == pytest.approx(240.0)
    assert result["total_co2_kg"] > 0
    assert "trees_equivalent" in result


def test_daily_carbon_estimate_partial_day() -> None:
    from app.carbon import daily_carbon_estimate

    result = daily_carbon_estimate([5.0] * 12)
    assert result["total_kwh"] == pytest.approx(60.0)


def test_daily_carbon_estimate_empty() -> None:
    from app.carbon import daily_carbon_estimate

    result = daily_carbon_estimate([])
    assert result["total_kwh"] == 0.0
    assert result["total_co2_kg"] == 0.0


def test_daily_carbon_estimate_too_many_hours_raises() -> None:
    from app.carbon import daily_carbon_estimate

    with pytest.raises(ValueError, match="24"):
        daily_carbon_estimate([1.0] * 25)


@pytest.mark.parametrize(
    "region,expected_gt",
    [
        ("pacific_nw", 0.0),
        ("midwest", 0.0),
        ("default", 0.0),
    ],
)
def test_daily_carbon_estimate_regions(region: str, expected_gt: float) -> None:
    from app.carbon import daily_carbon_estimate

    result = daily_carbon_estimate([10.0] * 8, region=region)
    assert result["total_co2_kg"] > expected_gt


def test_carbon_intensity_by_hour_basic() -> None:
    from app.carbon import carbon_intensity_by_hour

    result = carbon_intensity_by_hour([10.0, 20.0, 30.0])
    assert len(result) == 3
    assert result[0] < result[1] < result[2]


def test_carbon_intensity_by_hour_empty() -> None:
    from app.carbon import carbon_intensity_by_hour

    assert carbon_intensity_by_hour([]) == []


def test_carbon_intensity_by_hour_sums_to_daily() -> None:
    from app.carbon import carbon_intensity_by_hour, kwh_to_co2_kg

    hourly = [5.0] * 24
    result = carbon_intensity_by_hour(hourly)
    total = sum(result)
    expected = kwh_to_co2_kg(120.0)
    assert abs(total - expected) < 0.01


def test_carbon_intensity_by_hour_region() -> None:
    from app.carbon import carbon_intensity_by_hour

    default_result = carbon_intensity_by_hour([10.0])
    pacific_result = carbon_intensity_by_hour([10.0], region="pacific_nw")
    assert pacific_result[0] < default_result[0]


def test_tree_offset_days_basic() -> None:
    from app.carbon import tree_offset_days

    result = tree_offset_days(20.0, num_trees=1)
    assert result == pytest.approx(365.0, rel=0.01)


def test_tree_offset_days_zero_co2() -> None:
    from app.carbon import tree_offset_days

    assert tree_offset_days(0.0) == 0.0


def test_tree_offset_days_more_trees() -> None:
    from app.carbon import tree_offset_days

    single = tree_offset_days(100.0, num_trees=1)
    ten = tree_offset_days(100.0, num_trees=10)
    assert abs(single - 10 * ten) < 0.1


def test_tree_offset_days_invalid_trees_raises() -> None:
    from app.carbon import tree_offset_days

    with pytest.raises(ValueError, match="num_trees"):
        tree_offset_days(50.0, num_trees=0)


@pytest.mark.parametrize("co2,trees", [(10.0, 1), (10.0, 5), (365.0, 1)])
def test_tree_offset_days_parametrized(co2: float, trees: int) -> None:
    from app.carbon import tree_offset_days

    result = tree_offset_days(co2, num_trees=trees)
    assert result > 0


def test_monthly_co2_breakdown_length() -> None:
    from app.carbon import monthly_co2_breakdown

    result = monthly_co2_breakdown([100.0] * 12)
    assert len(result) == 12


def test_monthly_co2_breakdown_keys() -> None:
    from app.carbon import monthly_co2_breakdown

    result = monthly_co2_breakdown([100.0])
    assert "month" in result[0]
    assert "kwh" in result[0]
    assert "co2_kg" in result[0]


def test_monthly_co2_breakdown_month_index() -> None:
    from app.carbon import monthly_co2_breakdown

    result = monthly_co2_breakdown([10.0] * 3)
    assert result[0]["month"] == 1
    assert result[2]["month"] == 3


def test_monthly_co2_breakdown_empty() -> None:
    from app.carbon import monthly_co2_breakdown

    assert monthly_co2_breakdown([]) == []


def test_monthly_co2_breakdown_co2_proportional() -> None:
    from app.carbon import monthly_co2_breakdown

    result = monthly_co2_breakdown([100.0, 200.0])
    assert result[1]["co2_kg"] == pytest.approx(result[0]["co2_kg"] * 2, rel=1e-4)


class TestCarbonSavedKwh:
    def test_positive_savings(self) -> None:
        from app.carbon import carbon_saved_kwh

        result = carbon_saved_kwh(1000.0, 800.0)
        assert result["kwh_saved"] == pytest.approx(200.0, rel=1e-4)
        assert result["co2_kg_saved"] > 0
        assert result["pct_reduction"] == pytest.approx(20.0, rel=1e-4)

    def test_no_savings(self) -> None:
        from app.carbon import carbon_saved_kwh

        result = carbon_saved_kwh(500.0, 500.0)
        assert result["kwh_saved"] == pytest.approx(0.0, abs=1e-6)
        assert result["pct_reduction"] == pytest.approx(0.0, abs=1e-6)

    def test_negative_savings_overconsumption(self) -> None:
        from app.carbon import carbon_saved_kwh

        result = carbon_saved_kwh(100.0, 150.0)
        assert result["kwh_saved"] < 0
        assert result["pct_reduction"] < 0

    def test_zero_baseline_returns_zero_pct(self) -> None:
        from app.carbon import carbon_saved_kwh

        result = carbon_saved_kwh(0.0, 50.0)
        assert result["pct_reduction"] == 0.0

    def test_keys_present(self) -> None:
        from app.carbon import carbon_saved_kwh

        result = carbon_saved_kwh(200.0, 100.0)
        assert set(result.keys()) == {"kwh_saved", "co2_kg_saved", "pct_reduction"}

    def test_full_reduction(self) -> None:
        from app.carbon import carbon_saved_kwh

        result = carbon_saved_kwh(100.0, 0.0)
        assert result["pct_reduction"] == pytest.approx(100.0, rel=1e-4)
        assert result["kwh_saved"] == pytest.approx(100.0, rel=1e-4)


def test_kwh_to_co2_pacific_nw_lowest() -> None:
    from app.carbon import kwh_to_co2_kg

    pacific = kwh_to_co2_kg(100.0, "pacific_nw")
    midwest = kwh_to_co2_kg(100.0, "midwest")
    assert pacific < midwest


def test_co2_kg_to_tonnes_basic() -> None:
    from app.carbon import co2_kg_to_tonnes

    assert co2_kg_to_tonnes(1000.0) == pytest.approx(1.0)


def test_trees_equivalent_is_positive() -> None:
    from app.carbon import trees_equivalent

    result = trees_equivalent(1000.0)
    assert result > 0


@pytest.mark.parametrize("kwh", [0.0, 10.0, 100.0, 1000.0])
def test_kwh_to_co2_various_kwh(kwh: float) -> None:
    from app.carbon import kwh_to_co2_kg

    result = kwh_to_co2_kg(kwh)
    assert result >= 0.0
    assert result == pytest.approx(kwh * 0.35, rel=1e-3)


def test_tree_offset_days_positive_trees() -> None:
    from app.carbon import tree_offset_days

    result = tree_offset_days(365.0, num_trees=10)
    assert isinstance(result, float)
    assert result > 0


def test_monthly_co2_breakdown_returns_list() -> None:
    from app.carbon import monthly_co2_breakdown

    kwh_per_month = [100.0] * 12
    result = monthly_co2_breakdown(kwh_per_month)
    assert isinstance(result, list)
    assert len(result) == 12


def test_carbon_saved_kwh_returns_dict() -> None:
    from app.carbon import carbon_saved_kwh

    result = carbon_saved_kwh(100.0, 80.0)
    assert isinstance(result, dict)
    assert "co2_kg_saved" in result


def test_carbon_saved_kwh_positive_when_actual_less() -> None:
    from app.carbon import carbon_saved_kwh

    result = carbon_saved_kwh(100.0, 80.0)
    assert result["co2_kg_saved"] > 0


def test_annual_carbon_report_returns_dict() -> None:
    from app.carbon import annual_carbon_report

    result = annual_carbon_report([100.0] * 12)
    assert isinstance(result, dict)
    assert "total_co2_kg" in result


def test_annual_carbon_report_12_month_input() -> None:
    from app.carbon import annual_carbon_report

    result = annual_carbon_report([50.0] * 12)
    assert result["total_co2_kg"] > 0


@pytest.mark.parametrize("region", ["northeast", "west", "default"])
def test_carbon_saved_kwh_various_regions(region: str) -> None:
    from app.carbon import carbon_saved_kwh

    result = carbon_saved_kwh(200.0, 150.0, region=region)
    assert "co2_kg_saved" in result
    assert result["co2_kg_saved"] >= 0


class TestCompareRegions:
    def test_returns_list(self) -> None:
        from app.carbon import compare_regions

        result = compare_regions(100.0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_sorted_by_co2_ascending(self) -> None:
        from app.carbon import compare_regions

        result = compare_regions(100.0)
        co2_vals = [r["co2_kg"] for r in result]
        assert co2_vals == sorted(co2_vals)

    def test_custom_regions(self) -> None:
        from app.carbon import compare_regions

        result = compare_regions(50.0, regions=["northeast", "midwest"])
        assert len(result) == 2
        regions_in_result = {r["region"] for r in result}
        assert regions_in_result == {"northeast", "midwest"}

    def test_zero_kwh_gives_zero_co2(self) -> None:
        from app.carbon import compare_regions

        result = compare_regions(0.0, regions=["default"])
        assert result[0]["co2_kg"] == 0.0

    def test_negative_kwh_raises(self) -> None:
        import pytest

        from app.carbon import compare_regions

        with pytest.raises(ValueError, match="non-negative"):
            compare_regions(-1.0)

    @pytest.mark.parametrize("kwh", [1.0, 10.0, 500.0])
    def test_each_result_has_required_keys(self, kwh: float) -> None:
        from app.carbon import compare_regions

        for item in compare_regions(kwh, regions=["northeast", "west"]):
            assert "region" in item
            assert "co2_kg" in item
            assert "intensity_kg_per_kwh" in item


class TestCarbonBudgetRemaining:
    def test_within_budget(self) -> None:
        from app.carbon import carbon_budget_remaining

        result = carbon_budget_remaining(1000.0, 400.0)
        assert result["remaining_kg"] == pytest.approx(600.0)
        assert result["on_track"] == 1.0

    def test_over_budget(self) -> None:
        from app.carbon import carbon_budget_remaining

        result = carbon_budget_remaining(500.0, 600.0)
        assert result["remaining_kg"] == 0.0
        assert result["on_track"] == 0.0

    def test_used_pct(self) -> None:
        from app.carbon import carbon_budget_remaining

        result = carbon_budget_remaining(1000.0, 500.0)
        assert result["used_pct"] == pytest.approx(50.0)

    def test_zero_budget(self) -> None:
        from app.carbon import carbon_budget_remaining

        result = carbon_budget_remaining(0.0, 0.0)
        assert result["used_pct"] == 0.0

    def test_negative_budget_raises(self) -> None:
        from app.carbon import carbon_budget_remaining

        with pytest.raises(ValueError):
            carbon_budget_remaining(-100.0, 0.0)

    def test_negative_consumed_raises(self) -> None:
        from app.carbon import carbon_budget_remaining

        with pytest.raises(ValueError):
            carbon_budget_remaining(1000.0, -10.0)


class TestEmissionFactorChange:
    def test_cleaner_region_negative_change(self) -> None:
        from app.carbon import emission_factor_change

        result = emission_factor_change("midwest", "pacific_nw", 1000.0)
        assert result["co2_change_kg"] < 0

    def test_same_region_no_change(self) -> None:
        from app.carbon import emission_factor_change

        result = emission_factor_change("default", "default", 100.0)
        assert result["co2_change_kg"] == pytest.approx(0.0, abs=1e-6)

    def test_negative_kwh_raises(self) -> None:
        from app.carbon import emission_factor_change

        with pytest.raises(ValueError, match="non-negative"):
            emission_factor_change("default", "midwest", -100.0)

    def test_keys_present(self) -> None:
        from app.carbon import emission_factor_change

        result = emission_factor_change("northeast", "south", 500.0)
        assert set(result.keys()) >= {"old_co2_kg", "new_co2_kg", "co2_change_kg", "pct_change"}

    def test_zero_kwh(self) -> None:
        from app.carbon import emission_factor_change

        result = emission_factor_change("midwest", "west", 0.0)
        assert result["co2_change_kg"] == pytest.approx(0.0, abs=1e-6)


class TestCo2PerBuildingType:
    def test_basic(self) -> None:
        from app.carbon import co2_per_building_type

        result = co2_per_building_type(15.0, 1000.0)
        assert result["total_kwh"] == pytest.approx(15_000.0, rel=1e-4)
        assert result["total_co2_kg"] > 0

    def test_industrial_higher_than_residential(self) -> None:
        from app.carbon import co2_per_building_type

        industrial = co2_per_building_type(10.0, 1000.0, building_type="industrial")
        residential = co2_per_building_type(10.0, 1000.0, building_type="residential")
        assert industrial["total_co2_kg"] > residential["total_co2_kg"]

    def test_keys_present(self) -> None:
        from app.carbon import co2_per_building_type

        result = co2_per_building_type(10.0, 500.0, building_type="office")
        assert set(result.keys()) >= {"total_kwh", "total_co2_kg", "intensity_factor"}

    def test_unknown_type_uses_default(self) -> None:
        from app.carbon import co2_per_building_type

        result = co2_per_building_type(10.0, 1000.0, building_type="warehouse")
        assert result["intensity_factor"] == 1.0

    @pytest.mark.parametrize("btype", ["office", "retail", "industrial", "residential"])
    def test_known_types(self, btype: str) -> None:
        from app.carbon import co2_per_building_type

        result = co2_per_building_type(12.0, 2000.0, building_type=btype)
        assert result["total_kwh"] > 0


class TestCarbonScore:
    def test_zero_emissions_perfect_score(self) -> None:
        from app.carbon import carbon_score

        assert carbon_score(0.0, 1000.0) == pytest.approx(100.0, rel=1e-4)

    def test_max_emissions_zero_score(self) -> None:
        from app.carbon import carbon_score

        assert carbon_score(1000.0, 1000.0) == pytest.approx(0.0, abs=1e-6)

    def test_half_emissions(self) -> None:
        from app.carbon import carbon_score

        assert carbon_score(500.0, 1000.0) == pytest.approx(50.0, rel=1e-4)

    def test_negative_max_raises(self) -> None:
        from app.carbon import carbon_score

        with pytest.raises(ValueError, match="positive"):
            carbon_score(100.0, 0.0)

    def test_negative_co2_raises(self) -> None:
        from app.carbon import carbon_score

        with pytest.raises(ValueError, match="non-negative"):
            carbon_score(-1.0, 1000.0)

    def test_over_max_clamped_to_zero(self) -> None:
        from app.carbon import carbon_score

        assert carbon_score(2000.0, 1000.0) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("co2,max_kg", [(100.0, 1000.0), (0.0, 500.0), (250.0, 500.0)])
    def test_score_in_range(self, co2: float, max_kg: float) -> None:
        from app.carbon import carbon_score

        result = carbon_score(co2, max_kg)
        assert 0.0 <= result <= 100.0


class TestAnnualEmissionEstimate:
    def test_full_year(self) -> None:
        from app.carbon import annual_emission_estimate

        monthly = [100.0] * 12
        result = annual_emission_estimate(monthly, emission_factor=0.5)
        assert result == pytest.approx(600.0)

    def test_partial_year_extrapolates(self) -> None:
        from app.carbon import annual_emission_estimate

        monthly = [100.0] * 6
        result = annual_emission_estimate(monthly, emission_factor=0.5)
        assert result == pytest.approx(600.0)

    def test_empty_raises(self) -> None:
        from app.carbon import annual_emission_estimate

        with pytest.raises(ValueError):
            annual_emission_estimate([], 0.5)

    def test_invalid_factor_raises(self) -> None:
        from app.carbon import annual_emission_estimate

        with pytest.raises(ValueError):
            annual_emission_estimate([100.0], 0.0)


class TestCarbonReductionPotential:
    def test_positive_reduction(self) -> None:
        from app.carbon import carbon_reduction_potential

        result = carbon_reduction_potential(1000.0, 800.0, 0.5)
        assert result == pytest.approx(100.0)

    def test_zero_reduction(self) -> None:
        from app.carbon import carbon_reduction_potential

        result = carbon_reduction_potential(500.0, 500.0, 0.5)
        assert result == pytest.approx(0.0)

    def test_negative_when_increase(self) -> None:
        from app.carbon import carbon_reduction_potential

        result = carbon_reduction_potential(500.0, 700.0, 0.5)
        assert result < 0

    def test_invalid_factor_raises(self) -> None:
        from app.carbon import carbon_reduction_potential

        with pytest.raises(ValueError):
            carbon_reduction_potential(1000.0, 800.0, -0.1)


def test_lifetime_carbon_savings_basic() -> None:
    from app.carbon import lifetime_carbon_savings

    result = lifetime_carbon_savings(annual_kwh_saved=1000.0, lifetime_years=10)
    assert result["total_kwh_saved"] == pytest.approx(10_000.0)
    assert result["total_co2_kg_saved"] > 0


def test_lifetime_carbon_savings_keys() -> None:
    from app.carbon import lifetime_carbon_savings

    result = lifetime_carbon_savings(1000.0, 5)
    assert "total_kwh_saved" in result
    assert "total_co2_kg_saved" in result
    assert "total_co2_tonnes_saved" in result


def test_lifetime_carbon_savings_zero_years() -> None:
    from app.carbon import lifetime_carbon_savings

    result = lifetime_carbon_savings(1000.0, 0)
    assert result["total_kwh_saved"] == pytest.approx(0.0)
    assert result["total_co2_kg_saved"] == pytest.approx(0.0)


@pytest.mark.parametrize("years", [1, 5, 10, 25])
def test_lifetime_carbon_savings_scales_linearly(years) -> None:
    from app.carbon import lifetime_carbon_savings

    base = lifetime_carbon_savings(1000.0, 1)
    result = lifetime_carbon_savings(1000.0, years)
    assert result["total_kwh_saved"] == pytest.approx(base["total_kwh_saved"] * years)


def test_fleet_emission_factor_basic() -> None:
    from app.carbon import fleet_emission_factor

    fleet = [{"annual_kwh": 1000.0}, {"annual_kwh": 2000.0}]
    result = fleet_emission_factor(fleet)
    assert result["total_kwh"] == pytest.approx(3000.0)
    assert result["total_co2_kg"] > 0


def test_fleet_emission_factor_empty_fleet() -> None:
    from app.carbon import fleet_emission_factor

    result = fleet_emission_factor([])
    assert result["total_kwh"] == pytest.approx(0.0)
    assert result["mean_co2_kg_per_asset"] == pytest.approx(0.0)


def test_fleet_emission_factor_custom_region() -> None:
    from app.carbon import fleet_emission_factor

    fleet = [{"annual_kwh": 1000.0, "region": "pacific_nw"}]
    result = fleet_emission_factor(fleet)
    assert result["total_kwh"] == pytest.approx(1000.0)
    assert result["total_co2_kg"] > 0


@pytest.mark.parametrize("n_assets", [1, 3, 5])
def test_fleet_emission_factor_mean(n_assets) -> None:
    from app.carbon import fleet_emission_factor

    fleet = [{"annual_kwh": 1000.0} for _ in range(n_assets)]
    result = fleet_emission_factor(fleet)
    assert result["mean_co2_kg_per_asset"] == pytest.approx(result["total_co2_kg"] / n_assets, rel=1e-4)


def test_carbon_per_sqm_basic() -> None:
    from app.carbon import kwh_to_co2_kg

    result = carbon_per_sqm(10000.0, 100.0)
    expected = kwh_to_co2_kg(10000.0) / 100.0
    assert result == pytest.approx(expected, rel=1e-4)


def test_carbon_per_sqm_zero_area_raises() -> None:
    with pytest.raises(ValueError):
        carbon_per_sqm(1000.0, 0.0)


def test_carbon_per_sqm_negative_area_raises() -> None:
    with pytest.raises(ValueError):
        carbon_per_sqm(1000.0, -50.0)


def test_carbon_per_sqm_larger_area_smaller_result() -> None:
    small = carbon_per_sqm(1000.0, 100.0)
    large = carbon_per_sqm(1000.0, 200.0)
    assert small > large


def test_carbon_reduction_pct_positive() -> None:
    result = carbon_reduction_pct(100.0, 80.0)
    assert result == pytest.approx(20.0)


def test_carbon_reduction_pct_no_reduction() -> None:
    assert carbon_reduction_pct(100.0, 100.0) == pytest.approx(0.0)


def test_carbon_reduction_pct_zero_baseline() -> None:
    assert carbon_reduction_pct(0.0, 50.0) == pytest.approx(0.0)


def test_carbon_reduction_pct_negative_when_increased() -> None:
    result = carbon_reduction_pct(100.0, 120.0)
    assert result < 0.0


@pytest.mark.parametrize("region", ["northeast", "midwest", "west", "pacific_nw"])
def test_carbon_per_sqm_known_regions(region: str) -> None:
    result = carbon_per_sqm(5000.0, 100.0, region=region)
    assert result > 0.0
