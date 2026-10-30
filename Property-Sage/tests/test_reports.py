"""Tests for property_valuation_report."""
import pytest

from app.reports import property_valuation_report

SAMPLE_INPUT = {
    "bedrooms": 3, "bathrooms": 2.0, "sqft": 1500.0,
    "lot_size": 6000.0, "year_built": 2010,
    "neighborhood": "suburb", "property_type": "house",
}
SAMPLE_PRED = {
    "predicted_price": 450_000.0, "predicted_rental_yield": 0.05,
    "estimated_annual_rental": 22_500.0, "estimated_monthly_rental": 1_875.0,
}


def test_report_structure():
    r = property_valuation_report(SAMPLE_INPUT, SAMPLE_PRED)
    assert "summary" in r
    assert "input" in r
    assert "prediction" in r
    assert "drift" in r


def test_report_summary_fields():
    r = property_valuation_report(SAMPLE_INPUT, SAMPLE_PRED)
    s = r["summary"]
    assert "predicted_price" in s
    assert "rental_yield_pct" in s
    assert "yield_category" in s
    assert "price_per_sqft" in s
    assert "cap_rate" in s
    assert "property_age_category" in s


def test_report_drift_default():
    r = property_valuation_report(SAMPLE_INPUT, SAMPLE_PRED)
    assert r["drift"] == {"drift_detected": False}


def test_report_drift_passed():
    drift = {"drift_detected": True, "p_value": 0.03}
    r = property_valuation_report(SAMPLE_INPUT, SAMPLE_PRED, drift_result=drift)
    assert r["drift"]["drift_detected"] is True


@pytest.mark.parametrize("year,expected_cat", [
    (2020, "new"), (2009, "modern"), (1990, "established"),
])
def test_report_age_category(year, expected_cat):
    inp = dict(SAMPLE_INPUT, year_built=year)
    r = property_valuation_report(inp, SAMPLE_PRED)
    assert r["summary"]["property_age_category"] == expected_cat
