"""Tests for investment analysis module."""

import math

import pytest

from app.investment import InvestmentAnalysis, compute_investment_analysis


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
    for field in ("annual_rent_estimate", "cap_rate", "investment_score",
                  "break_even_years", "amenity_composite", "gross_rental_yield",
                  "risk_score", "predicted_value"):
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
    result = compute_investment_analysis(
        500_000, 0.08, 5, 5, 5, 0.2, operating_expense_ratio=expense_ratio
    )
    assert result.cap_rate == pytest.approx(
        (500_000 * 0.08 * (1 - expense_ratio)) / 500_000, rel=1e-4
    )
