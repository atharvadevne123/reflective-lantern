"""Tests for investment analysis module."""

import math

import pytest

from app.investment import (
    InvestmentAnalysis,
    break_even_occupancy,
    cash_on_cash_return,
    compute_investment_analysis,
    debt_service_coverage_ratio,
    discounted_cash_flow,
    equity_multiple,
    margin_of_safety,
    operating_expense_ratio,
    payback_period,
)


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


@pytest.mark.parametrize(
    "crime_rate,expected_worse",
    [
        (0.1, False),
        (0.9, True),
    ],
)
def test_compute_investment_high_crime_lowers_score(crime_rate, expected_worse) -> None:
    from app.investment import compute_investment_analysis

    low = compute_investment_analysis(
        predicted_value=400_000.0,
        avg_rental_yield=0.07,
        school_score=7.0,
        transit_score=6.0,
        walkability_score=5.0,
        crime_rate=0.1,
    )
    high = compute_investment_analysis(
        predicted_value=400_000.0,
        avg_rental_yield=0.07,
        school_score=7.0,
        transit_score=6.0,
        walkability_score=5.0,
        crime_rate=crime_rate,
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


def test_payback_period_basic() -> None:
    result = payback_period(10000.0, [5000.0, 5000.0, 5000.0])
    assert result == pytest.approx(2.0)


def test_payback_period_never_recovered() -> None:
    result = payback_period(10000.0, [100.0, 100.0])
    assert result == float("inf")


def test_payback_period_first_year() -> None:
    result = payback_period(1000.0, [2000.0, 500.0])
    assert result == pytest.approx(0.5)


def test_payback_period_zero_investment() -> None:
    result = payback_period(0.0, [100.0, 200.0])
    assert result == pytest.approx(0.0)


def test_payback_period_negative_investment_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        payback_period(-100.0, [50.0])


@pytest.mark.parametrize(
    "invest,flows,expected",
    [
        (300.0, [100.0, 100.0, 100.0], 3.0),
        (150.0, [100.0, 100.0], 1.5),
        (0.0, [1.0], 0.0),
    ],
)
def test_payback_period_parametrized(invest: float, flows: list, expected: float) -> None:
    result = payback_period(invest, flows)
    assert result == pytest.approx(expected, rel=1e-4)


class TestMarginOfSafety:
    def test_undervalued(self) -> None:
        # intrinsic 100, market 80 → 20% margin
        assert margin_of_safety(100.0, 80.0) == pytest.approx(20.0, rel=1e-4)

    def test_overvalued(self) -> None:
        # intrinsic 80, market 100 → -25% margin
        assert margin_of_safety(80.0, 100.0) == pytest.approx(-25.0, rel=1e-4)

    def test_at_fair_value(self) -> None:
        assert margin_of_safety(200.0, 200.0) == pytest.approx(0.0, abs=1e-9)

    def test_zero_intrinsic_returns_zero(self) -> None:
        assert margin_of_safety(0.0, 150.0) == 0.0

    def test_rounding_precision(self) -> None:
        result = margin_of_safety(300.0, 250.0)
        assert result == pytest.approx(16.6667, rel=1e-3)

    def test_large_values(self) -> None:
        result = margin_of_safety(1_000_000.0, 750_000.0)
        assert result == pytest.approx(25.0, rel=1e-4)


class TestIrrEstimate:
    def test_simple_positive_irr(self) -> None:
        from app.investment import irr_estimate

        irr = irr_estimate(1000.0, [200.0] * 8)
        assert irr > 0.0

    def test_negative_investment_raises(self) -> None:
        import pytest

        from app.investment import irr_estimate

        with pytest.raises(ValueError, match="positive"):
            irr_estimate(-100.0, [50.0])

    def test_no_return_gives_zero(self) -> None:
        from app.investment import irr_estimate

        irr = irr_estimate(1000.0, [0.0] * 5)
        assert irr == 0.0

    def test_with_terminal_value(self) -> None:
        from app.investment import irr_estimate

        irr = irr_estimate(1000.0, [50.0] * 5, terminal_value=1200.0)
        assert irr > 0.0

    def test_returns_float(self) -> None:
        from app.investment import irr_estimate

        result = irr_estimate(500.0, [100.0, 200.0, 300.0])
        assert isinstance(result, float)

    @pytest.mark.parametrize("flows", [[100.0, 200.0, 300.0], [50.0] * 10, [500.0, -100.0, 300.0]])
    def test_irr_returns_numeric(self, flows: list) -> None:
        from app.investment import irr_estimate

        result = irr_estimate(500.0, flows)
        assert isinstance(result, float)


class TestLoanToValueRatio:
    def test_basic(self) -> None:
        from app.investment import loan_to_value_ratio

        assert loan_to_value_ratio(160000, 200000) == pytest.approx(80.0)

    def test_full_loan(self) -> None:
        from app.investment import loan_to_value_ratio

        assert loan_to_value_ratio(200000, 200000) == pytest.approx(100.0)

    def test_zero_property_value(self) -> None:
        from app.investment import loan_to_value_ratio

        assert loan_to_value_ratio(100000, 0) == 0.0

    def test_zero_loan(self) -> None:
        from app.investment import loan_to_value_ratio

        assert loan_to_value_ratio(0, 200000) == pytest.approx(0.0)


class TestGrossRentMultiplier:
    def test_basic(self) -> None:
        from app.investment import gross_rent_multiplier

        assert gross_rent_multiplier(200000, 20000) == pytest.approx(10.0)

    def test_zero_rent_returns_inf(self) -> None:
        from app.investment import gross_rent_multiplier

        assert gross_rent_multiplier(200000, 0) == float("inf")

    def test_high_rent(self) -> None:
        from app.investment import gross_rent_multiplier

        result = gross_rent_multiplier(100000, 50000)
        assert result == pytest.approx(2.0)

    def test_negative_rent_returns_inf(self) -> None:
        from app.investment import gross_rent_multiplier

        assert gross_rent_multiplier(200000, -1000) == float("inf")


class TestAnnualizedReturn:
    def test_basic(self) -> None:
        from app.investment import annualized_return

        result = annualized_return(1.0, 10.0)
        assert result == pytest.approx((2.0) ** 0.1 - 1, rel=1e-4)

    def test_one_year(self) -> None:
        from app.investment import annualized_return

        assert annualized_return(0.10, 1.0) == pytest.approx(0.10, rel=1e-4)

    def test_zero_years_raises(self) -> None:
        from app.investment import annualized_return

        with pytest.raises(ValueError, match="positive"):
            annualized_return(0.5, 0.0)

    def test_total_return_minus_one_raises(self) -> None:
        from app.investment import annualized_return

        with pytest.raises(ValueError, match="> -1"):
            annualized_return(-1.0, 5.0)

    @pytest.mark.parametrize("total,years", [(0.0, 5.0), (0.5, 3.0), (2.0, 10.0)])
    def test_positive_return_positive_result(self, total: float, years: float) -> None:
        from app.investment import annualized_return

        assert annualized_return(total, years) >= 0.0


class TestNetPresentValue:
    def test_zero_discount(self) -> None:
        from app.investment import net_present_value

        flows = [-1000.0, 200.0, 300.0, 400.0, 500.0]
        result = net_present_value(flows, 0.0)
        assert result == pytest.approx(400.0, rel=1e-4)

    def test_positive_npv(self) -> None:
        from app.investment import net_present_value

        flows = [-100.0, 60.0, 60.0]
        result = net_present_value(flows, 0.10)
        assert result > 0

    def test_empty_raises(self) -> None:
        from app.investment import net_present_value

        with pytest.raises(ValueError, match="empty"):
            net_present_value([], 0.05)

    def test_discount_rate_below_minus_one_raises(self) -> None:
        from app.investment import net_present_value

        with pytest.raises(ValueError, match="> -1"):
            net_present_value([-100.0, 50.0], -2.0)

    def test_single_cash_flow(self) -> None:
        from app.investment import net_present_value

        assert net_present_value([-500.0], 0.10) == pytest.approx(-500.0, rel=1e-4)


class TestPropertyYieldAnalysis:
    def test_basic(self) -> None:
        from app.investment import property_yield_analysis

        result = property_yield_analysis(200_000.0, 20_000.0)
        assert result["gross_yield_pct"] == pytest.approx(10.0, rel=1e-4)

    def test_with_expenses(self) -> None:
        from app.investment import property_yield_analysis

        result = property_yield_analysis(200_000.0, 20_000.0, 5_000.0)
        assert result["net_yield_pct"] < result["gross_yield_pct"]

    def test_zero_market_value_raises(self) -> None:
        from app.investment import property_yield_analysis

        with pytest.raises(ValueError, match="positive"):
            property_yield_analysis(0.0, 12_000.0)

    def test_keys_present(self) -> None:
        from app.investment import property_yield_analysis

        result = property_yield_analysis(300_000.0, 18_000.0)
        assert set(result.keys()) >= {"gross_yield_pct", "net_yield_pct", "net_operating_income", "expense_ratio_pct"}


class TestEquityRatio:
    def test_no_loan(self) -> None:
        from app.investment import equity_ratio

        assert equity_ratio(200_000.0, 0.0) == pytest.approx(1.0, rel=1e-4)

    def test_fully_leveraged(self) -> None:
        from app.investment import equity_ratio

        assert equity_ratio(200_000.0, 200_000.0) == pytest.approx(0.0, abs=1e-4)

    def test_zero_market_value(self) -> None:
        from app.investment import equity_ratio

        assert equity_ratio(0.0, 0.0) == 0.0

    def test_negative_loan_raises(self) -> None:
        from app.investment import equity_ratio

        with pytest.raises(ValueError, match="non-negative"):
            equity_ratio(100_000.0, -5_000.0)

    def test_result_in_zero_one(self) -> None:
        from app.investment import equity_ratio

        result = equity_ratio(300_000.0, 120_000.0)
        assert 0.0 <= result <= 1.0


class TestEquityMultiple:
    def test_basic_em(self) -> None:
        assert equity_multiple(200_000.0, 100_000.0) == pytest.approx(2.0)

    def test_zero_invested_returns_zero(self) -> None:
        assert equity_multiple(100_000.0, 0.0) == 0.0

    def test_loss_scenario(self) -> None:
        assert equity_multiple(50_000.0, 100_000.0) == pytest.approx(0.5)

    def test_breakeven(self) -> None:
        assert equity_multiple(100_000.0, 100_000.0) == pytest.approx(1.0)


class TestCashOnCashReturn:
    def test_basic_coc(self) -> None:
        assert cash_on_cash_return(10_000.0, 100_000.0) == pytest.approx(10.0)

    def test_zero_invested_returns_zero(self) -> None:
        assert cash_on_cash_return(5_000.0, 0.0) == 0.0

    def test_negative_flow(self) -> None:
        result = cash_on_cash_return(-5_000.0, 100_000.0)
        assert result == pytest.approx(-5.0)

    @pytest.mark.parametrize(
        "flow,invested,expected",
        [
            (12_000.0, 120_000.0, 10.0),
            (6_000.0, 120_000.0, 5.0),
        ],
    )
    def test_coc_parametrize(self, flow, invested, expected) -> None:
        assert cash_on_cash_return(flow, invested) == pytest.approx(expected)


class TestDebtServiceCoverageRatio:
    def test_healthy_dscr(self) -> None:
        assert debt_service_coverage_ratio(120_000.0, 100_000.0) == pytest.approx(1.2)

    def test_zero_debt_service_returns_zero(self) -> None:
        assert debt_service_coverage_ratio(50_000.0, 0.0) == 0.0

    def test_below_one_dscr(self) -> None:
        result = debt_service_coverage_ratio(80_000.0, 100_000.0)
        assert result == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "noi,debt,expected",
        [
            (150_000.0, 100_000.0, 1.5),
            (100_000.0, 100_000.0, 1.0),
            (50_000.0, 100_000.0, 0.5),
        ],
    )
    def test_dscr_parametrize(self, noi, debt, expected) -> None:
        assert debt_service_coverage_ratio(noi, debt) == pytest.approx(expected)


class TestBreakEvenOccupancy:
    def test_basic(self) -> None:
        result = break_even_occupancy(30_000.0, 20_000.0, 100_000.0)
        assert result == pytest.approx(0.5)

    def test_zero_rent_returns_zero(self) -> None:
        assert break_even_occupancy(10_000.0, 5_000.0, 0.0) == pytest.approx(0.0)

    def test_above_one_when_unviable(self) -> None:
        result = break_even_occupancy(80_000.0, 30_000.0, 50_000.0)
        assert result > 1.0

    def test_perfectly_covered(self) -> None:
        result = break_even_occupancy(50_000.0, 50_000.0, 100_000.0)
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "opex,debt,gross",
        [
            (20_000.0, 10_000.0, 100_000.0),
            (40_000.0, 20_000.0, 200_000.0),
        ],
    )
    def test_parametrize(self, opex: float, debt: float, gross: float) -> None:
        result = break_even_occupancy(opex, debt, gross)
        assert result == pytest.approx((opex + debt) / gross)


class TestOperatingExpenseRatio:
    def test_basic(self) -> None:
        assert operating_expense_ratio(30_000.0, 100_000.0) == pytest.approx(30.0)

    def test_zero_egi_returns_zero(self) -> None:
        assert operating_expense_ratio(10_000.0, 0.0) == pytest.approx(0.0)

    def test_full_expense(self) -> None:
        assert operating_expense_ratio(100_000.0, 100_000.0) == pytest.approx(100.0)

    def test_low_expense(self) -> None:
        result = operating_expense_ratio(10_000.0, 100_000.0)
        assert result == pytest.approx(10.0)


class TestEquityMultipleAdditional:
    def test_basic(self) -> None:
        result = equity_multiple(total_profit=150_000.0, equity_invested=100_000.0)
        assert result == pytest.approx(1.5)

    def test_zero_equity_returns_zero(self) -> None:
        assert equity_multiple(total_profit=50_000.0, equity_invested=0.0) == pytest.approx(0.0)

    def test_no_profit_returns_one(self) -> None:
        result = equity_multiple(total_profit=100_000.0, equity_invested=100_000.0)
        assert result == pytest.approx(1.0)


class TestMarginOfSafetyAdditional:
    def test_positive_margin(self) -> None:
        result = margin_of_safety(intrinsic_value=200_000.0, market_price=150_000.0)
        assert result == pytest.approx(25.0)

    def test_no_margin(self) -> None:
        result = margin_of_safety(intrinsic_value=100_000.0, market_price=100_000.0)
        assert result == pytest.approx(0.0)

    def test_negative_margin_when_overpriced(self) -> None:
        result = margin_of_safety(intrinsic_value=100_000.0, market_price=120_000.0)
        assert result < 0.0


@pytest.mark.parametrize(
    "purchase_price,annual_cash_flow,expected_years",
    [
        (100_000.0, 10_000.0, 10.0),
        (50_000.0, 25_000.0, 2.0),
        (100_000.0, 0.0, float("inf")),
    ],
)
def test_payback_period_new_parametrized(purchase_price: float, annual_cash_flow: float, expected_years: float) -> None:
    result = payback_period(purchase_price=purchase_price, annual_cash_flow=annual_cash_flow)
    if math.isinf(expected_years):
        assert math.isinf(result)
    else:
        assert result == pytest.approx(expected_years)


class TestPriceToRentRatio:
    def test_basic(self) -> None:
        from app.investment import price_to_rent_ratio

        assert price_to_rent_ratio(300_000.0, 15_000.0) == pytest.approx(20.0)

    def test_zero_rent_returns_zero(self) -> None:
        from app.investment import price_to_rent_ratio

        assert price_to_rent_ratio(300_000.0, 0.0) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "price,rent,expected",
        [
            (200_000.0, 10_000.0, 20.0),
            (150_000.0, 15_000.0, 10.0),
        ],
    )
    def test_parametrize(self, price: float, rent: float, expected: float) -> None:
        from app.investment import price_to_rent_ratio

        assert price_to_rent_ratio(price, rent) == pytest.approx(expected)


class TestHoldingPeriodReturn:
    def test_no_income(self) -> None:
        from app.investment import holding_period_return

        assert holding_period_return(100_000.0, 120_000.0) == pytest.approx(20.0)

    def test_with_income(self) -> None:
        from app.investment import holding_period_return

        assert holding_period_return(100_000.0, 100_000.0, total_income=10_000.0) == pytest.approx(10.0)

    def test_loss(self) -> None:
        from app.investment import holding_period_return

        assert holding_period_return(100_000.0, 80_000.0) == pytest.approx(-20.0)

    def test_zero_purchase_raises(self) -> None:
        from app.investment import holding_period_return

        with pytest.raises(ValueError):
            holding_period_return(0.0, 100_000.0)


class TestRentToValueRatio:
    def test_basic_ratio(self) -> None:
        from app.investment import rent_to_value_ratio

        assert rent_to_value_ratio(12_000.0, 200_000.0) == pytest.approx(6.0)

    def test_zero_property_value_returns_zero(self) -> None:
        from app.investment import rent_to_value_ratio

        assert rent_to_value_ratio(10_000.0, 0.0) == pytest.approx(0.0)

    def test_negative_rent_raises(self) -> None:
        from app.investment import rent_to_value_ratio

        with pytest.raises(ValueError):
            rent_to_value_ratio(-1.0, 100_000.0)

    def test_negative_value_raises(self) -> None:
        from app.investment import rent_to_value_ratio

        with pytest.raises(ValueError):
            rent_to_value_ratio(10_000.0, -1.0)

    @pytest.mark.parametrize(
        "rent,value,expected",
        [
            (24_000.0, 200_000.0, 12.0),
            (10_000.0, 100_000.0, 10.0),
            (0.0, 100_000.0, 0.0),
        ],
    )
    def test_parametrized(self, rent, value, expected) -> None:
        from app.investment import rent_to_value_ratio

        assert rent_to_value_ratio(rent, value) == pytest.approx(expected)


class TestEffectiveGrossIncome:
    def test_no_vacancy(self) -> None:
        from app.investment import effective_gross_income

        assert effective_gross_income(100_000.0, vacancy_rate=0.0) == pytest.approx(100_000.0)

    def test_five_percent_vacancy(self) -> None:
        from app.investment import effective_gross_income

        assert effective_gross_income(100_000.0, vacancy_rate=0.05) == pytest.approx(95_000.0)

    def test_other_income_added(self) -> None:
        from app.investment import effective_gross_income

        result = effective_gross_income(100_000.0, vacancy_rate=0.0, other_income=5_000.0)
        assert result == pytest.approx(105_000.0)

    def test_invalid_vacancy_raises(self) -> None:
        from app.investment import effective_gross_income

        with pytest.raises(ValueError):
            effective_gross_income(100_000.0, vacancy_rate=1.5)

    def test_negative_income_raises(self) -> None:
        from app.investment import effective_gross_income

        with pytest.raises(ValueError):
            effective_gross_income(-1.0)


class TestCapitalisationRate:
    def test_basic_cap_rate(self) -> None:
        from app.investment import capitalisation_rate

        assert capitalisation_rate(10_000.0, 200_000.0) == pytest.approx(5.0)

    def test_zero_value_returns_zero(self) -> None:
        from app.investment import capitalisation_rate

        assert capitalisation_rate(10_000.0, 0.0) == pytest.approx(0.0)

    def test_negative_value_raises(self) -> None:
        from app.investment import capitalisation_rate

        with pytest.raises(ValueError):
            capitalisation_rate(1_000.0, -50_000.0)

    @pytest.mark.parametrize(
        "noi,value,expected",
        [
            (20_000.0, 200_000.0, 10.0),
            (5_000.0, 100_000.0, 5.0),
        ],
    )
    def test_parametrized(self, noi, value, expected) -> None:
        from app.investment import capitalisation_rate

        assert capitalisation_rate(noi, value) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests for gross_yield, price_to_rent_ratio, equity_multiple
# ---------------------------------------------------------------------------


class TestGrossYield:
    def test_basic(self) -> None:
        from app.investment import gross_yield

        assert gross_yield(12000.0, 200000.0) == pytest.approx(6.0, abs=0.01)

    def test_zero_property_value(self) -> None:
        from app.investment import gross_yield

        assert gross_yield(12000.0, 0.0) == 0.0

    def test_negative_raises(self) -> None:
        import pytest

        from app.investment import gross_yield

        with pytest.raises(ValueError):
            gross_yield(-1.0, 200000.0)

    def test_zero_rent(self) -> None:
        from app.investment import gross_yield

        assert gross_yield(0.0, 200000.0) == 0.0


class TestPriceToRentRatioNew:
    def test_standard_case(self) -> None:
        from app.investment import price_to_rent_ratio

        result = price_to_rent_ratio(300000.0, 1250.0)
        assert result == pytest.approx(20.0, abs=0.1)

    def test_zero_rent(self) -> None:
        from app.investment import price_to_rent_ratio

        assert price_to_rent_ratio(300000.0, 0.0) == 0.0

    def test_negative_raises(self) -> None:
        import pytest

        from app.investment import price_to_rent_ratio

        with pytest.raises(ValueError):
            price_to_rent_ratio(-100.0, 1000.0)


class TestEquityMultipleNew:
    def test_double_money(self) -> None:
        from app.investment import equity_multiple

        assert equity_multiple(200000.0, 100000.0) == 2.0

    def test_zero_invested(self) -> None:
        from app.investment import equity_multiple

        assert equity_multiple(1000.0, 0.0) == 0.0

    def test_negative_raises(self) -> None:
        import pytest

        from app.investment import equity_multiple

        with pytest.raises(ValueError):
            equity_multiple(-1.0, 100000.0)

    def test_loss_scenario(self) -> None:
        from app.investment import equity_multiple

        result = equity_multiple(50000.0, 100000.0)
        assert result < 1.0


import pytest as _pytest


@_pytest.mark.parametrize(
    "noi,debt_service,expect_healthy",
    [
        (100000.0, 80000.0, True),
        (80000.0, 100000.0, False),
        (100000.0, 100000.0, True),
    ],
)
def test_debt_service_coverage_ratio_parametrized(noi: float, debt_service: float, expect_healthy: bool) -> None:
    from app.investment import debt_service_coverage_ratio

    result = debt_service_coverage_ratio(noi, debt_service)
    assert (result >= 1.0) == expect_healthy


@_pytest.mark.parametrize(
    "loan,market_value,expected",
    [
        (400000.0, 500000.0, _pytest.approx(80.0, abs=0.001)),
        (0.0, 500000.0, _pytest.approx(0.0, abs=0.001)),
        (500000.0, 500000.0, _pytest.approx(100.0, abs=0.001)),
    ],
)
def test_loan_to_value_ratio_parametrized(loan: float, market_value: float, expected: float) -> None:
    from app.investment import loan_to_value_ratio

    assert loan_to_value_ratio(loan, market_value) == expected


@_pytest.mark.parametrize("rate", [0.05, 0.10, 0.15])
def test_net_present_value_positive_for_positive_flows(rate: float) -> None:
    from app.investment import net_present_value

    cash_flows = [-1000.0] + [300.0] * 5
    npv = net_present_value(cash_flows, rate)
    assert isinstance(npv, float)


@_pytest.mark.parametrize("score,expected_label", [(90.0, "Excellent"), (70.0, "Good"), (50.0, "Fair"), (30.0, "Poor")])
def test_investment_label_thresholds(score: float, expected_label: str) -> None:
    from app.investment import investment_score_label

    assert investment_score_label(score) == expected_label


@_pytest.mark.parametrize(
    "property_price,rent,expected_range",
    [
        (100000.0, 1000.0, (5.0, 200.0)),
        (500000.0, 2000.0, (5.0, 500.0)),
    ],
)
def test_price_to_rent_ratio_in_range(property_price: float, rent: float, expected_range: tuple) -> None:
    from app.investment import price_to_rent_ratio

    result = price_to_rent_ratio(property_price, rent)
    assert expected_range[0] <= result <= expected_range[1]


@_pytest.mark.parametrize(
    "purchase_price,sale_price,income,expected_positive",
    [
        (100.0, 150.0, 10.0, True),
        (100.0, 80.0, 0.0, False),
    ],
)
def test_holding_period_return_sign(purchase_price: float, sale_price: float, income: float, expected_positive: bool) -> None:
    from app.investment import holding_period_return

    result = holding_period_return(purchase_price, sale_price, income)
    if expected_positive:
        assert result > 0.0
    else:
        assert result < 0.0


@_pytest.mark.parametrize("annual_rent,property_value", [(12000.0, 200000.0), (24000.0, 400000.0)])
def test_gross_yield_in_range(annual_rent: float, property_value: float) -> None:
    from app.investment import gross_yield

    result = gross_yield(annual_rent, property_value)
    assert 0.0 < result < 100.0


