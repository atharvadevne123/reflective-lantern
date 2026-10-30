"""Tests for property_valuation_report."""

import pytest

from app.reports import property_valuation_report

SAMPLE_INPUT = {
    "bedrooms": 3, "bathrooms": 2.0, "sqft": 1500.0,
    "lot_size": 6000.0, "year_built": 2010,
    "neighborhood": "suburb", "property_type": "house",
}
SAMPLE_PREDICTION = {
    "predicted_price": 450_000.0,
    "predicted_rental_yield": 0.05,
    "estimated_annual_rental": 22_500.0,
    "estimated_monthly_rental": 1_875.0,
}


def test_report_structure():
    report = property_valuation_report(SAMPLE_INPUT, SAMPLE_PREDICTION)
    assert "summary" in report
    assert "input" in report
    assert "prediction" in report
    assert "drift" in report


def test_report_summary_fields():
    report = property_valuation_report(SAMPLE_INPUT, SAMPLE_PREDICTION)
    summary = report["summary"]
    assert "predicted_price_formatted" in summary
    assert "yield_category" in summary
    assert "cap_rate" in summary


def test_report_drift_default():
    report = property_valuation_report(SAMPLE_INPUT, SAMPLE_PREDICTION)
    assert report["drift"]["drift_detected"] is False


@pytest.mark.parametrize("year,expected_cat", [
    (2020, "new"), (2005, "modern"), (1990, "established")
])
def test_report_age_category(year, expected_cat):
    inp = {**SAMPLE_INPUT, "year_built": year}
    report = property_valuation_report(inp, SAMPLE_PREDICTION)
    assert report["summary"]["property_age_category"] == expected_cat
