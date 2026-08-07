"""Tests for market context enrichment functions."""

import math

import pytest

from app.market_context import (
    affordability_index,
    comparable_value_adjustment,
    dom_classification,
    effective_gross_income,
    market_heat_score,
    price_appreciation_rate,
    price_per_bedroom,
    price_per_sqft,
    price_to_rent_ratio,
    value_gap,
)


def test_price_per_sqft_basic() -> None:
    assert price_per_sqft(500_000, 1000) == pytest.approx(500.0)


def test_price_per_sqft_zero_sqft() -> None:
    assert price_per_sqft(500_000, 0) == 0.0


@pytest.mark.parametrize(
    "days,expected",
    [
        (5, "fast"),
        (13, "fast"),
        (14, "normal"),
        (60, "normal"),
        (61, "slow"),
        (365, "slow"),
    ],
)
def test_dom_classification(days, expected) -> None:
    assert dom_classification(days) == expected


def test_affordability_expensive_is_not_affordable() -> None:
    result = affordability_index(predicted_value=2_000_000, annual_income=80_000)
    assert result["is_affordable"] is False
    assert result["pct_income"] > 28


def test_affordability_cheap_is_affordable() -> None:
    result = affordability_index(predicted_value=150_000, annual_income=100_000)
    assert result["is_affordable"] is True


def test_affordability_contains_required_keys() -> None:
    result = affordability_index(500_000)
    assert "loan_amount" in result
    assert "monthly_payment" in result
    assert "pct_income" in result
    assert "is_affordable" in result


def test_affordability_loan_amount() -> None:
    result = affordability_index(500_000, down_payment_pct=0.20)
    assert result["loan_amount"] == pytest.approx(400_000.0)


def test_price_to_rent_ratio_basic() -> None:
    result = price_to_rent_ratio(500_000, 25_000)
    assert result == pytest.approx(20.0)


def test_price_to_rent_ratio_zero_rent() -> None:
    result = price_to_rent_ratio(500_000, 0)
    assert math.isinf(result)


@pytest.mark.parametrize(
    "value,rent,expect_buy",
    [
        (200_000, 20_000, True),
        (1_000_000, 20_000, False),
    ],
)
def test_buy_vs_rent_signal(value, rent, expect_buy) -> None:
    ratio = price_to_rent_ratio(value, rent)
    assert (ratio < 15) == expect_buy


@pytest.mark.parametrize(
    "sqft,expected",
    [
        (1000.0, 200.0),
        (500.0, 400.0),
        (2000.0, 100.0),
    ],
)
def test_price_per_sqft_parametrized(sqft, expected) -> None:
    from app.market_context import price_per_sqft

    assert price_per_sqft(200_000, sqft) == pytest.approx(expected)


@pytest.mark.parametrize(
    "days,expected",
    [
        (5, "fast"),
        (13, "fast"),
        (14, "normal"),
        (60, "normal"),
        (61, "slow"),
        (365, "slow"),
    ],
)
def test_dom_classification_boundary_values(days, expected) -> None:
    from app.market_context import dom_classification

    assert dom_classification(days) == expected


def test_price_to_rent_ratio_zero_annual_rent_returns_inf() -> None:
    from app.market_context import price_to_rent_ratio

    ratio = price_to_rent_ratio(500_000, 0)
    assert ratio == float("inf")


def test_affordability_index_high_price_not_affordable() -> None:
    from app.market_context import affordability_index

    result = affordability_index(2_000_000, annual_income=80_000)
    assert result["is_affordable"] is False


def test_affordability_index_low_price_is_affordable() -> None:
    from app.market_context import affordability_index

    result = affordability_index(100_000, annual_income=200_000)
    assert result["is_affordable"] is True


def test_affordability_bucket_affordable() -> None:
    from app.market_context import affordability_bucket

    assert affordability_bucket(25.0) == "affordable"
    assert affordability_bucket(28.0) == "affordable"


def test_affordability_bucket_stretched() -> None:
    from app.market_context import affordability_bucket

    assert affordability_bucket(28.1) == "stretched"
    assert affordability_bucket(36.0) == "stretched"


def test_affordability_bucket_unaffordable() -> None:
    from app.market_context import affordability_bucket

    assert affordability_bucket(36.1) == "unaffordable"
    assert affordability_bucket(80.0) == "unaffordable"


@pytest.mark.parametrize(
    "pct,expected",
    [
        (0.0, "affordable"),
        (28.0, "affordable"),
        (28.5, "stretched"),
        (36.0, "stretched"),
        (36.1, "unaffordable"),
        (100.0, "unaffordable"),
    ],
)
def test_affordability_bucket_parametrized(pct: float, expected: str) -> None:
    from app.market_context import affordability_bucket

    assert affordability_bucket(pct) == expected


@pytest.mark.parametrize(
    "value,sqft,expected_ppsf",
    [
        (300_000, 1_500, 200.0),
        (600_000, 2_000, 300.0),
        (1_000_000, 4_000, 250.0),
        (500_000, 1_000, 500.0),
    ],
)
def test_price_per_sqft_various_inputs(value: float, sqft: float, expected_ppsf: float) -> None:
    assert price_per_sqft(value, sqft) == pytest.approx(expected_ppsf)


@pytest.mark.parametrize(
    "days,bucket",
    [
        (0, "fast"),
        (1, "fast"),
        (13, "fast"),
        (14, "normal"),
        (30, "normal"),
        (60, "normal"),
        (61, "slow"),
        (180, "slow"),
        (730, "slow"),
    ],
)
def test_dom_classification_full_range(days: int, bucket: str) -> None:
    assert dom_classification(days) == bucket


def test_affordability_monthly_payment_positive() -> None:
    result = affordability_index(predicted_value=400_000, annual_income=100_000)
    assert result["monthly_payment"] > 0


def test_price_to_rent_ratio_high_rent_low_ratio() -> None:
    ratio = price_to_rent_ratio(300_000, 60_000)
    assert ratio == pytest.approx(5.0)


def test_affordability_pct_income_positive() -> None:
    result = affordability_index(predicted_value=500_000, annual_income=100_000)
    assert result["pct_income"] > 0


def test_dom_fast_threshold_constant() -> None:
    from app.market_context import DOM_FAST_DAYS

    assert dom_classification(DOM_FAST_DAYS - 1) == "fast"
    assert dom_classification(DOM_FAST_DAYS) == "normal"


def test_dom_slow_threshold_constant() -> None:
    from app.market_context import DOM_SLOW_DAYS

    assert dom_classification(DOM_SLOW_DAYS) == "normal"
    assert dom_classification(DOM_SLOW_DAYS + 1) == "slow"


def test_affordability_threshold_constant() -> None:
    from app.market_context import AFFORDABILITY_THRESHOLD, affordability_bucket

    assert affordability_bucket(AFFORDABILITY_THRESHOLD) == "affordable"
    assert affordability_bucket(AFFORDABILITY_THRESHOLD + 0.1) == "stretched"


def test_stretched_threshold_constant() -> None:
    from app.market_context import STRETCHED_THRESHOLD, affordability_bucket

    assert affordability_bucket(STRETCHED_THRESHOLD) == "stretched"
    assert affordability_bucket(STRETCHED_THRESHOLD + 0.1) == "unaffordable"


@pytest.mark.parametrize(
    "income,expected_loan_fraction",
    [
        (100_000.0, 0.80),
        (200_000.0, 0.80),
    ],
)
def test_affordability_loan_equals_80pct_at_default_down_payment(income: float, expected_loan_fraction: float) -> None:
    value = 500_000.0
    result = affordability_index(value, annual_income=income)
    assert result["loan_amount"] == pytest.approx(value * expected_loan_fraction)


@pytest.mark.parametrize(
    "ptr,expected",
    [
        (10.0, "buy"),
        (18.0, "neutral"),
        (25.0, "rent"),
    ],
)
def test_rent_vs_buy_recommendation(ptr, expected) -> None:
    from app.market_context import rent_vs_buy_comparison

    # ptr = price / annual_rent so annual_rent = price / ptr
    price = 400_000
    annual_rent = price / ptr
    result = rent_vs_buy_comparison(price, annual_rent)
    assert result["recommendation"] == expected


def test_rent_vs_buy_has_required_keys() -> None:
    from app.market_context import rent_vs_buy_comparison

    result = rent_vs_buy_comparison(400_000, 20_000)
    assert "monthly_rent" in result
    assert "monthly_mortgage" in result
    assert "rent_premium" in result
    assert "price_to_rent_ratio" in result
    assert "recommendation" in result


def test_price_trend_rising() -> None:
    from app.market_context import price_trend_indicator

    result = price_trend_indicator(420_000, 400_000)
    assert result["trend"] == "rising"
    assert result["absolute_change"] == pytest.approx(20_000.0)


def test_price_trend_falling() -> None:
    from app.market_context import price_trend_indicator

    result = price_trend_indicator(380_000, 400_000)
    assert result["trend"] == "falling"


def test_price_trend_stable() -> None:
    from app.market_context import price_trend_indicator

    result = price_trend_indicator(401_000, 400_000)
    assert result["trend"] == "stable"


def test_price_trend_zero_previous() -> None:
    from app.market_context import price_trend_indicator

    result = price_trend_indicator(400_000, 0)
    assert result["trend"] == "unknown"


def test_hai_affordable() -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(300_000, 100_000, mortgage_rate=0.065, term_years=30)
    assert result["hai"] > 100  # affordable


def test_hai_unaffordable() -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(1_500_000, 80_000, mortgage_rate=0.07)
    assert result["hai"] < 100  # unaffordable


def test_hai_zero_income() -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(300_000, 0)
    assert result["hai"] == 0.0


def test_hai_zero_price() -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(0, 100_000)
    assert result["hai"] == 0.0


def test_hai_has_required_keys() -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(400_000, 90_000)
    for key in ("loan_amount", "monthly_payment", "qualifying_income", "hai"):
        assert key in result


@pytest.mark.parametrize("down_pct", [0.05, 0.10, 0.20, 0.30])
def test_hai_higher_down_lowers_payment(down_pct) -> None:
    from app.market_context import housing_affordability_index

    full = housing_affordability_index(400_000, 80_000, down_payment_pct=0.0)
    result = housing_affordability_index(400_000, 80_000, down_payment_pct=down_pct)
    if down_pct > 0:
        assert result["monthly_payment"] < full["monthly_payment"]


def test_affordability_ratio_basic() -> None:
    from app.market_context import affordability_ratio

    ratio = affordability_ratio(300_000, 100_000, down_payment_pct=0.20)
    assert ratio == pytest.approx(2.4, rel=1e-3)


def test_affordability_ratio_zero_income() -> None:
    from app.market_context import affordability_ratio

    assert affordability_ratio(300_000, 0) == 0.0


def test_affordability_ratio_no_down_payment() -> None:
    from app.market_context import affordability_ratio

    ratio = affordability_ratio(200_000, 100_000, down_payment_pct=0.0)
    assert ratio == pytest.approx(2.0, rel=1e-3)


def test_affordability_ratio_full_down_payment() -> None:
    from app.market_context import affordability_ratio

    ratio = affordability_ratio(200_000, 100_000, down_payment_pct=1.0)
    assert ratio == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "price,income,down,expected_max",
    [
        (200_000, 100_000, 0.0, 2.1),
        (500_000, 100_000, 0.2, 4.1),
        (150_000, 80_000, 0.1, 1.7),
    ],
)
def test_affordability_ratio_parametrized(price, income, down, expected_max) -> None:
    from app.market_context import affordability_ratio

    ratio = affordability_ratio(price, income, down_payment_pct=down)
    assert ratio <= expected_max


def test_price_trend_consistency_rising() -> None:
    from app.market_context import price_trend_consistency

    prices = [100_000, 110_000, 120_000, 130_000, 140_000]
    result = price_trend_consistency(prices)
    assert result["trend"] == "rising"
    assert result["consistency_score"] == pytest.approx(1.0)


def test_price_trend_consistency_falling() -> None:
    from app.market_context import price_trend_consistency

    prices = [200_000, 190_000, 180_000, 170_000]
    result = price_trend_consistency(prices)
    assert result["trend"] == "falling"


def test_price_trend_consistency_mixed() -> None:
    from app.market_context import price_trend_consistency

    prices = [100_000, 110_000, 105_000, 115_000, 110_000]
    result = price_trend_consistency(prices)
    assert result["direction_changes"] > 0


def test_price_trend_consistency_too_few_raises() -> None:
    from app.market_context import price_trend_consistency

    with pytest.raises(ValueError):
        price_trend_consistency([100_000])


def test_price_trend_consistency_keys() -> None:
    from app.market_context import price_trend_consistency

    result = price_trend_consistency([1.0, 2.0, 3.0])
    assert set(result.keys()) >= {"direction_changes", "consistency_score", "trend", "periods"}


@pytest.mark.parametrize(
    "prices,expected_trend",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], "rising"),
        ([5.0, 4.0, 3.0, 2.0, 1.0], "falling"),
    ],
)
def test_price_trend_consistency_parametrized(prices, expected_trend) -> None:
    from app.market_context import price_trend_consistency

    result = price_trend_consistency(prices)
    assert result["trend"] == expected_trend


def test_market_summary_keys() -> None:
    from app.market_context import market_summary

    result = market_summary(500_000, 30_000, 20, 1500)
    assert set(result.keys()) >= {"price_per_sqft", "dom", "dom_classification", "price_to_rent_ratio", "affordability"}


def test_market_summary_price_per_sqft() -> None:
    from app.market_context import market_summary

    result = market_summary(500_000, 30_000, 20, 1000)
    assert result["price_per_sqft"] == pytest.approx(500.0)


def test_market_summary_dom_classification_fast() -> None:
    from app.market_context import market_summary

    result = market_summary(400_000, 24_000, 5, 1200)
    assert result["dom_classification"] == "fast"


def test_market_summary_dom_classification_slow() -> None:
    from app.market_context import market_summary

    result = market_summary(400_000, 24_000, 90, 1200)
    assert result["dom_classification"] == "slow"


def test_market_summary_ptr_ratio() -> None:
    from app.market_context import market_summary

    result = market_summary(300_000, 20_000, 30, 1200)
    assert result["price_to_rent_ratio"] == pytest.approx(15.0)


def test_market_summary_zero_sqft() -> None:
    from app.market_context import market_summary

    result = market_summary(500_000, 30_000, 30, 0)
    assert result["price_per_sqft"] == 0.0


@pytest.mark.parametrize(
    "listing_days,expected_dom",
    [
        (10, "fast"),
        (30, "normal"),
        (75, "slow"),
    ],
)
def test_market_summary_dom_parametrized(listing_days, expected_dom) -> None:
    from app.market_context import market_summary

    result = market_summary(400_000, 24_000, listing_days, 1000)
    assert result["dom_classification"] == expected_dom


def test_affordability_index_default_income() -> None:
    from app.market_context import affordability_index

    result = affordability_index(300_000.0)
    assert isinstance(result, dict)
    assert result["monthly_payment"] > 0


def test_affordability_index_zero_value() -> None:
    from app.market_context import affordability_index

    result = affordability_index(0.0)
    assert result["monthly_payment"] == 0.0


@pytest.mark.parametrize(
    "value,income,expected_range",
    [
        (100_000, 100_000, (0, 20)),  # very affordable
        (1_000_000, 50_000, (100, 500)),  # very expensive
    ],
)
def test_affordability_index_parametrized(value, income, expected_range) -> None:
    from app.market_context import affordability_index

    result = affordability_index(value, annual_income=income)
    lo, hi = expected_range
    assert lo <= result["pct_income"] <= hi


def test_price_to_rent_ratio_extended() -> None:
    from app.market_context import price_to_rent_ratio

    result = price_to_rent_ratio(300_000.0, 20_000.0)
    assert abs(result - 15.0) < 0.01


def test_price_to_rent_ratio_zero_rent_v2() -> None:
    import math

    from app.market_context import price_to_rent_ratio

    result = price_to_rent_ratio(300_000.0, 0.0)
    assert math.isinf(result)


class TestNeighbourhoodScore:
    def test_returns_composite_and_grade(self) -> None:
        from app.market_context import neighbourhood_score

        result = neighbourhood_score(8.0, 7.0, 9.0)
        assert "composite_score" in result
        assert "grade" in result

    def test_high_scores_give_grade_a(self) -> None:
        from app.market_context import neighbourhood_score

        result = neighbourhood_score(9.0, 9.0, 9.0, safety_score=9.0)
        assert result["grade"] == "A"

    def test_low_scores_give_grade_d(self) -> None:
        from app.market_context import neighbourhood_score

        result = neighbourhood_score(1.0, 1.0, 1.0, safety_score=1.0)
        assert result["grade"] == "D"

    def test_clamping_above_10(self) -> None:
        from app.market_context import neighbourhood_score

        result = neighbourhood_score(15.0, 15.0, 15.0, safety_score=15.0)
        assert result["composite_score"] <= 10.0

    def test_clamping_below_0(self) -> None:
        from app.market_context import neighbourhood_score

        result = neighbourhood_score(-5.0, -5.0, -5.0, safety_score=-5.0)
        assert result["composite_score"] >= 0.0

    def test_custom_weights_invalid_raises(self) -> None:
        import pytest

        from app.market_context import neighbourhood_score

        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            neighbourhood_score(
                7.0, 7.0, 7.0, weights={"school": 0.5, "transit": 0.5, "walkability": 0.5, "safety": 0.5}
            )

    @pytest.mark.parametrize("school,transit,walk", [(5, 5, 5), (8, 3, 6), (10, 10, 10)])
    def test_score_within_bounds(self, school: float, transit: float, walk: float) -> None:
        from app.market_context import neighbourhood_score

        result = neighbourhood_score(school, transit, walk)
        assert 0.0 <= result["composite_score"] <= 10.0


class TestPriceGrowthRate:
    def test_positive_growth(self) -> None:
        from app.market_context import price_growth_rate

        result = price_growth_rate(100_000.0, 121_000.0, years=2.0)
        assert result == pytest.approx(10.0, rel=1e-3)

    def test_no_growth(self) -> None:
        from app.market_context import price_growth_rate

        assert price_growth_rate(100_000.0, 100_000.0) == pytest.approx(0.0, abs=1e-4)

    def test_negative_growth(self) -> None:
        from app.market_context import price_growth_rate

        result = price_growth_rate(100_000.0, 90_000.0)
        assert result < 0.0

    def test_zero_old_price_raises(self) -> None:
        from app.market_context import price_growth_rate

        with pytest.raises(ValueError, match="positive"):
            price_growth_rate(0.0, 100_000.0)

    def test_zero_years_raises(self) -> None:
        from app.market_context import price_growth_rate

        with pytest.raises(ValueError, match="positive"):
            price_growth_rate(100_000.0, 120_000.0, years=0.0)

    @pytest.mark.parametrize("old,new,yrs", [(100.0, 200.0, 10.0), (500.0, 750.0, 5.0)])
    def test_returns_float(self, old: float, new: float, yrs: float) -> None:
        from app.market_context import price_growth_rate

        assert isinstance(price_growth_rate(old, new, yrs), float)


class TestMarketHeatIndex:
    def test_hot_market(self) -> None:
        from app.market_context import market_heat_index

        assert market_heat_index(10, 60.0) == "hot"

    def test_warm_market(self) -> None:
        from app.market_context import market_heat_index

        assert market_heat_index(20, 30.0) == "warm"

    def test_neutral_market(self) -> None:
        from app.market_context import market_heat_index

        assert market_heat_index(45, 10.0) == "neutral"

    def test_cool_market(self) -> None:
        from app.market_context import market_heat_index

        assert market_heat_index(90, 5.0) == "cool"

    @pytest.mark.parametrize(
        "dom,pct,expected",
        [
            (5, 75, "hot"),
            (25, 40, "warm"),
            (50, 5, "neutral"),
            (120, 2, "cool"),
        ],
    )
    def test_parametrized(self, dom: int, pct: float, expected: str) -> None:
        from app.market_context import market_heat_index

        assert market_heat_index(dom, pct) == expected


class TestVacancyAdjustedYield:
    def test_zero_vacancy(self) -> None:
        from app.market_context import vacancy_adjusted_yield

        assert vacancy_adjusted_yield(6.0, 0.0) == pytest.approx(6.0, rel=1e-4)

    def test_full_vacancy(self) -> None:
        from app.market_context import vacancy_adjusted_yield

        assert vacancy_adjusted_yield(6.0, 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_partial_vacancy(self) -> None:
        from app.market_context import vacancy_adjusted_yield

        assert vacancy_adjusted_yield(10.0, 0.1) == pytest.approx(9.0, rel=1e-4)

    def test_invalid_rate_raises(self) -> None:
        from app.market_context import vacancy_adjusted_yield

        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            vacancy_adjusted_yield(5.0, 1.5)

    @pytest.mark.parametrize("yield_pct,vac", [(8.0, 0.05), (6.0, 0.0), (10.0, 0.2)])
    def test_result_non_negative(self, yield_pct: float, vac: float) -> None:
        from app.market_context import vacancy_adjusted_yield

        assert vacancy_adjusted_yield(yield_pct, vac) >= 0.0


class TestNormalizeMarketFeatures:
    def test_basic(self) -> None:
        from app.market_context import normalize_market_features

        result = normalize_market_features({"a": 0.0, "b": 50.0, "c": 100.0})
        assert result["a"] == pytest.approx(0.0, abs=1e-6)
        assert result["c"] == pytest.approx(1.0, rel=1e-4)

    def test_empty_raises(self) -> None:
        from app.market_context import normalize_market_features

        with pytest.raises(ValueError, match="empty"):
            normalize_market_features({})

    def test_constant_values_return_half(self) -> None:
        from app.market_context import normalize_market_features

        result = normalize_market_features({"a": 5.0, "b": 5.0, "c": 5.0})
        assert all(v == 0.5 for v in result.values())

    def test_keys_preserved(self) -> None:
        from app.market_context import normalize_market_features

        features = {"school": 8.0, "transit": 4.0, "walk": 6.0}
        result = normalize_market_features(features)
        assert set(result.keys()) == set(features.keys())

    def test_values_in_range(self) -> None:
        from app.market_context import normalize_market_features

        result = normalize_market_features({"a": 1.0, "b": 5.0, "c": 3.0})
        assert all(0.0 <= v <= 1.0 for v in result.values())


def test_price_per_bedroom_basic() -> None:
    assert price_per_bedroom(400_000.0, 4) == pytest.approx(100_000.0)


def test_price_per_bedroom_zero_bedrooms() -> None:
    assert price_per_bedroom(400_000.0, 0) == 0.0


def test_price_per_bedroom_negative_bedrooms() -> None:
    assert price_per_bedroom(400_000.0, -1) == 0.0


@pytest.mark.parametrize(
    "price,beds,expected",
    [
        (300_000.0, 3, 100_000.0),
        (200_000.0, 2, 100_000.0),
        (500_000.0, 5, 100_000.0),
    ],
)
def test_price_per_bedroom_parametrize(price, beds, expected) -> None:
    assert price_per_bedroom(price, beds) == pytest.approx(expected)


def test_market_heat_score_hot_market() -> None:
    score = market_heat_score(days_on_market=5, list_to_sale_ratio=1.05, inventory_months=0.5)
    assert score > 7.0


def test_market_heat_score_cold_market() -> None:
    score = market_heat_score(days_on_market=90, list_to_sale_ratio=0.90, inventory_months=12.0)
    assert score < 3.0


def test_market_heat_score_range() -> None:
    score = market_heat_score(30, 1.0, 3.0)
    assert 0.0 <= score <= 10.0


def test_value_gap_underpriced() -> None:
    gap = value_gap(estimated_value=300_000.0, list_price=250_000.0)
    assert gap > 0


def test_value_gap_overpriced() -> None:
    gap = value_gap(estimated_value=200_000.0, list_price=250_000.0)
    assert gap < 0


def test_value_gap_zero_list_price() -> None:
    assert value_gap(300_000.0, 0.0) == 0.0


def test_value_gap_at_fair_value() -> None:
    assert value_gap(300_000.0, 300_000.0) == pytest.approx(0.0)


def test_comparable_value_adjustment_larger_subject() -> None:
    result = comparable_value_adjustment(
        subject_sqft=1500.0, comp_sqft=1000.0, comp_price=200_000.0, price_per_sqft_adj=100.0
    )
    assert result == pytest.approx(250_000.0)


def test_comparable_value_adjustment_smaller_subject() -> None:
    result = comparable_value_adjustment(
        subject_sqft=800.0, comp_sqft=1000.0, comp_price=200_000.0, price_per_sqft_adj=100.0
    )
    assert result == pytest.approx(180_000.0)


def test_comparable_value_adjustment_equal_sqft() -> None:
    result = comparable_value_adjustment(subject_sqft=1000.0, comp_sqft=1000.0, comp_price=200_000.0)
    assert result == pytest.approx(200_000.0)


@pytest.mark.parametrize(
    "dom,ratio,inv,hot",
    [
        (5, 1.05, 0.5, True),
        (90, 0.90, 12.0, False),
    ],
)
def test_market_heat_score_parametrize(dom, ratio, inv, hot) -> None:
    score = market_heat_score(dom, ratio, inv)
    if hot:
        assert score >= 5.0
    else:
        assert score <= 5.0


def test_egi_no_vacancy() -> None:
    assert effective_gross_income(100_000.0, vacancy_rate=0.0) == pytest.approx(100_000.0)


def test_egi_with_5pct_vacancy() -> None:
    assert effective_gross_income(100_000.0, vacancy_rate=0.05) == pytest.approx(95_000.0)


def test_egi_full_vacancy() -> None:
    assert effective_gross_income(100_000.0, vacancy_rate=1.0) == pytest.approx(0.0)


def test_egi_invalid_vacancy_raises() -> None:
    with pytest.raises(ValueError):
        effective_gross_income(100_000.0, vacancy_rate=1.1)


def test_egi_negative_vacancy_raises() -> None:
    with pytest.raises(ValueError):
        effective_gross_income(100_000.0, vacancy_rate=-0.1)


def test_price_appreciation_rate_flat() -> None:
    assert price_appreciation_rate(200_000.0, 200_000.0, 5.0) == pytest.approx(0.0)


def test_price_appreciation_rate_positive() -> None:
    rate = price_appreciation_rate(100_000.0, 121_000.0, 2.0)
    assert rate == pytest.approx(10.0, rel=1e-3)


def test_price_appreciation_rate_negative() -> None:
    rate = price_appreciation_rate(200_000.0, 150_000.0, 5.0)
    assert rate < 0.0


def test_price_appreciation_rate_zero_original_raises() -> None:
    with pytest.raises(ValueError):
        price_appreciation_rate(0.0, 200_000.0, 5.0)


def test_price_appreciation_rate_zero_years_raises() -> None:
    with pytest.raises(ValueError):
        price_appreciation_rate(100_000.0, 200_000.0, 0.0)


@pytest.mark.parametrize("vacancy", [0.0, 0.05, 0.10, 0.20])
def test_egi_parametrize(vacancy: float) -> None:
    gross = 120_000.0
    result = effective_gross_income(gross, vacancy_rate=vacancy)
    assert result == pytest.approx(gross * (1.0 - vacancy))


def test_housing_affordability_index_above_100_when_affordable() -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(
        median_home_price=200_000.0,
        median_household_income=100_000.0,
        mortgage_rate=0.04,
    )
    assert result["hai"] > 100.0


def test_housing_affordability_index_zero_income_returns_zero() -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(
        median_home_price=200_000.0,
        median_household_income=0.0,
    )
    assert result["hai"] == 0.0


def test_housing_affordability_index_has_required_keys() -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(
        median_home_price=300_000.0,
        median_household_income=80_000.0,
    )
    assert set(result.keys()) >= {"loan_amount", "monthly_payment", "qualifying_income", "hai"}


@pytest.mark.parametrize(
    "price,income,expected_affordable",
    [
        (200_000.0, 150_000.0, True),   # very affordable
        (1_000_000.0, 50_000.0, False),  # not affordable
    ],
)
def test_housing_affordability_index_parametrized(
    price: float, income: float, expected_affordable: bool
) -> None:
    from app.market_context import housing_affordability_index

    result = housing_affordability_index(median_home_price=price, median_household_income=income)
    assert (result["hai"] >= 100) is expected_affordable
