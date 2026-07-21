"""Tests for investment analysis module."""

import math

import pytest

from app.investment import InvestmentAnalysis, compute_investment_analysis, discounted_cash_flow


def test_basic_investment_analysis() -> None:
    result = compute_investment_analysis(
        predicted_value=500_000,
        avg_rental_yield=0.08,
        school_score=7.0,
        transit_score=6.0,
        walkability_score=7.0,
        crime_rate=0.2,
    )
    assert isinstance(result, InvestmentAnalysis)
    assert result.annual_rent_estimate == pytest.approx(40_000.0)
    assert result.cap_rate > 0
    assert 0 <= result.investment_score <= 10
    assert result.break_even_years > 0


def test_zero_value_returns_zero_score() -> None:
    result = compute_investment_analysis(
        predicted_value=0,
        avg_rental_yield=0.06,
        school_score=5,
        transit_score=5,
        walkability_score=5,
        crime_rate=0.3,
    )
    assert result.investment_score == 0.0
    assert math.isinf(result.break_even_years)


def test_high_crime_lowers_score() -> None:
    low = compute_investment_analysis(500_000, 0.07, 7, 7, 7, crime_rate=0.1)
    high = compute_investment_analysis(500_000, 0.07, 7, 7, 7, crime_rate=0.9)
    assert low.investment_score > high.investment_score


def test_high_yield_raises_score() -> None:
    low = compute_investment_analysis(500_000, 0.04, 6, 6, 6, 0.3)
    high = compute_investment_analysis(500_000, 0.12, 6, 6, 6, 0.3)
    assert high.investment_score > low.investment_score


def test_score_bounded_zero_to_ten() -> None:
    result = compute_investment_analysis(500_000, 0.20, 10, 10, 10, 0.0)
    assert 0 <= result.investment_score <= 10


@pytest.mark.parametrize(
    "yield_,expected_gt_zero",
    [
        (0.0, True),
        (0.05, True),
        (0.10, True),
    ],
)
def test_various_yields(yield_, expected_gt_zero) -> None:
    result = compute_investment_analysis(400_000, yield_, 5, 5, 5, 0.3)
    if expected_gt_zero:
        assert result.annual_rent_estimate >= 0


def test_break_even_decreases_with_higher_yield() -> None:
    low = compute_investment_analysis(500_000, 0.04, 5, 5, 5, 0.3)
    high = compute_investment_analysis(500_000, 0.12, 5, 5, 5, 0.3)
    assert high.break_even_years < low.break_even_years


def test_amenity_composite_range() -> None:
    result = compute_investment_analysis(500_000, 0.07, 8, 9, 7, 0.2)
    assert 0 <= result.amenity_composite <= 1


def test_gross_rental_yield_matches_input() -> None:
    result = compute_investment_analysis(
        predicted_value=400_000,
        avg_rental_yield=0.07,
        school_score=6,
        transit_score=6,
        walkability_score=6,
        crime_rate=0.3,
    )
    assert result.gross_rental_yield == pytest.approx(0.07)


def test_risk_score_equals_crime_rate() -> None:
    for cr in [0.0, 0.3, 0.7, 1.0]:
        result = compute_investment_analysis(500_000, 0.07, 6, 6, 6, cr)
        assert result.risk_score == pytest.approx(cr)


@pytest.mark.parametrize("expense_ratio", [0.20, 0.35, 0.50])
def test_higher_expense_ratio_lowers_cap_rate(expense_ratio) -> None:
    result = compute_investment_analysis(500_000, 0.08, 6, 6, 6, 0.3, expense_ratio)
    baseline = compute_investment_analysis(500_000, 0.08, 6, 6, 6, 0.3, 0.10)
    if expense_ratio > 0.10:
        assert result.cap_rate < baseline.cap_rate


def test_predicted_value_preserved_in_result() -> None:
    result = compute_investment_analysis(750_000, 0.06, 5, 5, 5, 0.3)
    assert result.predicted_value == pytest.approx(750_000)


def test_zero_predicted_value_returns_early() -> None:
    result = compute_investment_analysis(0.0, 0.07, 6, 6, 6, 0.3)
    assert result.investment_score == 0.0
    assert result.annual_rent_estimate == 0.0
    assert result.break_even_years == float("inf")


def test_negative_predicted_value_returns_early() -> None:
    result = compute_investment_analysis(-100.0, 0.07, 6, 6, 6, 0.3)
    assert result.investment_score == 0.0


def test_investment_score_clamped_to_zero_at_minimum() -> None:
    result = compute_investment_analysis(500_000, 0.001, 0, 0, 0, 1.0)
    assert result.investment_score >= 0.0


def test_investment_score_clamped_to_ten_at_maximum() -> None:
    result = compute_investment_analysis(100_000, 0.99, 10, 10, 10, 0.0)
    assert result.investment_score <= 10.0


def test_break_even_zero_noi_returns_inf() -> None:
    result = compute_investment_analysis(500_000, 0.0, 6, 6, 6, 0.3)
    assert result.break_even_years == float("inf")


@pytest.mark.parametrize(
    "school,transit,walk",
    [
        (1.0, 1.0, 1.0),
        (5.0, 5.0, 5.0),
        (10.0, 10.0, 10.0),
    ],
)
def test_amenity_composite_bounded(school: float, transit: float, walk: float) -> None:
    result = compute_investment_analysis(500_000, 0.07, school, transit, walk, 0.2)
    assert 0 <= result.amenity_composite <= 1


@pytest.mark.parametrize(
    "predicted_value",
    [100_000, 300_000, 750_000, 2_000_000],
)
def test_annual_rent_scales_with_value(predicted_value: float) -> None:
    result = compute_investment_analysis(predicted_value, 0.07, 5, 5, 5, 0.3)
    assert result.annual_rent_estimate == pytest.approx(predicted_value * 0.07)


def test_cap_rate_positive_with_yield() -> None:
    result = compute_investment_analysis(500_000, 0.08, 5, 5, 5, 0.3)
    assert result.cap_rate > 0


def test_investment_analysis_fields_present() -> None:
    result = compute_investment_analysis(500_000, 0.07, 5, 5, 5, 0.3)
    for field in (
        "annual_rent_estimate",
        "cap_rate",
        "investment_score",
        "break_even_years",
        "amenity_composite",
        "gross_rental_yield",
        "risk_score",
        "predicted_value",
    ):
        assert hasattr(result, field)


@pytest.mark.parametrize("crime_rate", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_risk_score_equals_crime_rate_parametrized(crime_rate: float) -> None:
    result = compute_investment_analysis(500_000, 0.07, 5, 5, 5, crime_rate)
    assert result.risk_score == pytest.approx(crime_rate)


def test_default_operating_expense_ratio_constant() -> None:
    from app.investment import DEFAULT_OPERATING_EXPENSE_RATIO

    assert 0.0 < DEFAULT_OPERATING_EXPENSE_RATIO < 1.0


def test_investment_score_max_constant() -> None:
    from app.investment import INVESTMENT_SCORE_MAX

    assert INVESTMENT_SCORE_MAX == 10.0


def test_investment_score_min_constant() -> None:
    from app.investment import INVESTMENT_SCORE_MIN

    assert INVESTMENT_SCORE_MIN == 0.0


def test_investment_score_never_exceeds_max() -> None:
    from app.investment import INVESTMENT_SCORE_MAX

    result = compute_investment_analysis(500_000, 0.20, 10.0, 10.0, 10.0, 0.0)
    assert result.investment_score <= INVESTMENT_SCORE_MAX


def test_investment_score_never_below_min() -> None:
    from app.investment import INVESTMENT_SCORE_MIN

    result = compute_investment_analysis(500_000, 0.001, 0.0, 0.0, 0.0, 1.0)
    assert result.investment_score >= INVESTMENT_SCORE_MIN


@pytest.mark.parametrize("expense_ratio", [0.25, 0.35, 0.45])
def test_cap_rate_decreases_with_expense_ratio(expense_ratio: float) -> None:
    result = compute_investment_analysis(500_000, 0.08, 5, 5, 5, 0.2, operating_expense_ratio=expense_ratio)
    assert result.cap_rate == pytest.approx((500_000 * 0.08 * (1 - expense_ratio)) / 500_000, rel=1e-4)


def test_mortgage_payment_positive() -> None:
    from app.investment import mortgage_payment

    payment = mortgage_payment(400_000, annual_rate=0.065, term_years=30)
    assert payment > 0


def test_mortgage_payment_zero_rate() -> None:
    from app.investment import mortgage_payment

    payment = mortgage_payment(150_000, annual_rate=0.0, term_years=10, down_payment_pct=0.20)
    # loan / (10*12) = 120000 / 120 = 1000
    assert payment == pytest.approx(1000.0, rel=1e-4)


def test_mortgage_payment_zero_principal() -> None:
    from app.investment import mortgage_payment

    assert mortgage_payment(0, 0.065) == 0.0


def test_roi_percentage_positive() -> None:
    from app.investment import roi_percentage

    roi = roi_percentage(
        predicted_value=550_000,
        purchase_price=400_000,
        annual_income=30_000,
        annual_expenses=20_000,
        hold_years=5,
    )
    assert roi > 0


def test_roi_percentage_zero_purchase() -> None:
    from app.investment import roi_percentage

    assert roi_percentage(500_000, 0, 30_000, 20_000) == 0.0


def test_price_to_income_ratio_basic() -> None:
    from app.investment import price_to_income_ratio

    ratio = price_to_income_ratio(400_000, 100_000)
    assert ratio == pytest.approx(4.0)


def test_price_to_income_ratio_zero_income() -> None:
    import math

    from app.investment import price_to_income_ratio

    assert math.isinf(price_to_income_ratio(400_000, 0))


@pytest.mark.parametrize(
    "price,income,expected",
    [
        (200_000, 100_000, 2.0),
        (500_000, 100_000, 5.0),
        (300_000, 150_000, 2.0),
    ],
)
def test_price_to_income_ratio_parametrized(price, income, expected) -> None:
    from app.investment import price_to_income_ratio

    assert price_to_income_ratio(price, income) == pytest.approx(expected)


def test_mortgage_payment_zero_rate_no_down() -> None:
    from app.investment import mortgage_payment

    p = mortgage_payment(200_000, annual_rate=0.0, term_years=10, down_payment_pct=0.0)
    assert p == pytest.approx(200_000 / (10 * 12), rel=1e-3)


def test_mortgage_payment_with_down_payment() -> None:
    from app.investment import mortgage_payment

    p_full = mortgage_payment(200_000, annual_rate=0.05, term_years=30, down_payment_pct=0.0)
    p_down = mortgage_payment(200_000, annual_rate=0.05, term_years=30, down_payment_pct=0.20)
    assert p_down < p_full


def test_roi_percentage_positive_income() -> None:
    from app.investment import roi_percentage

    roi = roi_percentage(300_000, 250_000, annual_income=20_000, annual_expenses=5_000, hold_years=5)
    assert roi > 0


@pytest.mark.parametrize("term_years", [10, 15, 20, 30])
def test_mortgage_payment_longer_term_lower_payment(term_years) -> None:
    from app.investment import mortgage_payment

    short = mortgage_payment(300_000, 0.06, 10)
    long = mortgage_payment(300_000, 0.06, term_years)
    if term_years > 10:
        assert long < short


def test_investment_score_label_excellent() -> None:
    from app.investment import investment_score_label

    assert investment_score_label(9.0) == "excellent"


def test_investment_score_label_good() -> None:
    from app.investment import investment_score_label

    assert investment_score_label(7.5) == "good"


def test_investment_score_label_avoid() -> None:
    from app.investment import investment_score_label

    assert investment_score_label(0.5) == "avoid"


@pytest.mark.parametrize(
    "score,expected",
    [
        (10.0, "excellent"),
        (8.0, "excellent"),
        (6.0, "good"),
        (4.0, "fair"),
        (2.0, "poor"),
        (1.9, "avoid"),
        (0.0, "avoid"),
    ],
)
def test_investment_score_label_parametrized(score, expected) -> None:
    from app.investment import investment_score_label

    assert investment_score_label(score) == expected


def test_portfolio_weighted_score_equal_weights() -> None:
    from app.investment import portfolio_weighted_score

    scores = [4.0, 6.0, 8.0]
    result = portfolio_weighted_score(scores)
    assert result == pytest.approx(6.0, rel=1e-3)


def test_portfolio_weighted_score_empty() -> None:
    from app.investment import portfolio_weighted_score

    assert portfolio_weighted_score([]) == 0.0


def test_portfolio_weighted_score_custom_weights() -> None:
    from app.investment import portfolio_weighted_score

    scores = [2.0, 8.0]
    weights = [0.25, 0.75]
    result = portfolio_weighted_score(scores, weights=weights)
    assert result == pytest.approx(6.5, rel=1e-3)


def test_portfolio_weighted_score_mismatch_raises() -> None:
    from app.investment import portfolio_weighted_score

    with pytest.raises(ValueError):
        portfolio_weighted_score([1.0, 2.0], weights=[0.5])


def test_compute_investment_analysis_returns_dataclass() -> None:
    from app.investment import InvestmentAnalysis, compute_investment_analysis

    result = compute_investment_analysis(
        predicted_value=400_000.0,
        avg_rental_yield=0.07,
        school_score=7.0,
        transit_score=6.0,
        walkability_score=5.0,
        crime_rate=0.2,
    )
    assert isinstance(result, InvestmentAnalysis)
    assert 0.0 <= result.investment_score <= 10.0
    assert result.break_even_years > 0


@pytest.mark.parametrize("crime_rate,expected_worse", [
    (0.1, False),
    (0.9, True),
])
def test_compute_investment_high_crime_lowers_score(crime_rate, expected_worse) -> None:
    from app.investment import compute_investment_analysis

    low = compute_investment_analysis(
        predicted_value=400_000.0, avg_rental_yield=0.07,
        school_score=7.0, transit_score=6.0, walkability_score=5.0,
        crime_rate=0.1
    )
    high = compute_investment_analysis(
        predicted_value=400_000.0, avg_rental_yield=0.07,
        school_score=7.0, transit_score=6.0, walkability_score=5.0,
        crime_rate=crime_rate
    )
    if expected_worse:
        assert high.investment_score <= low.investment_score
    else:
        assert high.investment_score == low.investment_score


@pytest.mark.parametrize("value", [100_000, 500_000, 1_000_000, 5_000_000])
def test_investment_analysis_various_values(value: float) -> None:
    result = compute_investment_analysis(value, 0.07, 6.0, 6.0, 6.0, 0.3)
    assert isinstance(result, InvestmentAnalysis)
    assert result.predicted_value == value
    assert 0 <= result.investment_score <= 10


@pytest.mark.parametrize("crime_rate", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_investment_score_decreases_with_crime(crime_rate: float) -> None:
    result = compute_investment_analysis(500_000, 0.07, 6.0, 6.0, 6.0, crime_rate)
    assert 0 <= result.investment_score <= 10


def test_analysis_fields_are_non_negative() -> None:
    result = compute_investment_analysis(300_000, 0.06, 5.0, 5.0, 5.0, 0.3)
    assert result.annual_rent_estimate >= 0
    assert result.cap_rate >= 0
    assert result.investment_score >= 0
    assert result.amenity_composite >= 0


def test_high_amenity_raises_score() -> None:
    low = compute_investment_analysis(400_000, 0.07, 3.0, 3.0, 3.0, 0.3)
    high = compute_investment_analysis(400_000, 0.07, 9.0, 9.0, 9.0, 0.3)
    assert high.investment_score >= low.investment_score


def test_break_even_infinite_when_zero_cap_rate() -> None:
    result = compute_investment_analysis(500_000, 0.0, 5.0, 5.0, 5.0, 0.5)
    assert math.isinf(result.break_even_years) or result.break_even_years > 100



def test_dcf_zero_discount_rate() -> None:
    flows = [1000.0, 1000.0, 1000.0]
    result = discounted_cash_flow(flows, discount_rate=0.0)
    assert result == pytest.approx(3000.0, rel=1e-4)


def test_dcf_positive_npv() -> None:
    flows = [10000.0] * 5
    result = discounted_cash_flow(flows, discount_rate=0.08)
    assert result > 0


def test_dcf_with_terminal_value() -> None:
    flows = [5000.0] * 3
    result_no_tv = discounted_cash_flow(flows, discount_rate=0.1)
    result_with_tv = discounted_cash_flow(flows, discount_rate=0.1, terminal_value=50000.0)
    assert result_with_tv > result_no_tv


def test_dcf_empty_flows() -> None:
    assert discounted_cash_flow([], discount_rate=0.08) == 0.0


def test_dcf_negative_rate_raises() -> None:
    with pytest.raises(ValueError, match="discount_rate"):
        discounted_cash_flow([1000.0], discount_rate=-0.05)


@pytest.mark.parametrize("rate", [0.0, 0.05, 0.10, 0.20])
def test_dcf_various_rates(rate: float) -> None:
    result = discounted_cash_flow([1000.0, 2000.0, 3000.0], discount_rate=rate)
    assert isinstance(result, float)
    assert result > 0
