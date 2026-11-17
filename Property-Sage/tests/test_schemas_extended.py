"""Extended tests for Property-Sage Pydantic response schemas."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import (
    DriftCheckResponse,
    HealthResponse,
    MetricsResponse,
    NeighbourhoodReportResponse,
    NeighbourhoodSearchResponse,
    PredictionResponse,
)


def test_health_response_fields():
    h = HealthResponse(status="ok", service="property-sage", version="1.0.0")
    assert h.status == "ok"
    assert h.service == "property-sage"
    assert h.version == "1.0.0"


def test_health_response_missing_field_raises():
    with pytest.raises(ValidationError):
        HealthResponse(status="ok", service="property-sage")


def test_prediction_response_all_fields():
    pr = PredictionResponse(
        request_id="req-001",
        predicted_price=350000.0,
        predicted_rental_yield=0.05,
        estimated_annual_rental=17500.0,
        estimated_monthly_rental=1458.33,
        neighborhood="suburb",
        property_type="house",
    )
    assert pr.predicted_price == 350000.0
    assert pr.neighborhood == "suburb"


def test_prediction_response_request_id():
    pr = PredictionResponse(
        request_id="req-abc",
        predicted_price=100000.0,
        predicted_rental_yield=0.04,
        estimated_annual_rental=4000.0,
        estimated_monthly_rental=333.33,
        neighborhood="downtown",
        property_type="condo",
    )
    assert pr.request_id == "req-abc"


def test_drift_check_response_defaults():
    dcr = DriftCheckResponse(status="ok")
    assert dcr.drift_detected is None
    assert dcr.checks == []


def test_drift_check_response_with_drift():
    dcr = DriftCheckResponse(status="drift", drift_detected=True, checks=[{"feature": "sqft"}])
    assert dcr.drift_detected is True
    assert len(dcr.checks) == 1


def test_metrics_response_defaults():
    mr = MetricsResponse()
    assert isinstance(mr.model_performance, dict)
    assert isinstance(mr.prediction_stats, dict)


def test_metrics_response_custom_dicts():
    mr = MetricsResponse(
        model_performance={"rmse": 15000.0},
        prediction_stats={"count": 100},
    )
    assert mr.model_performance["rmse"] == 15000.0


def test_neighbourhood_report_response():
    nr = NeighbourhoodReportResponse(
        neighbourhood="downtown", market_report="Strong demand in downtown."
    )
    assert nr.neighbourhood == "downtown"
    assert "Strong demand" in nr.market_report


def test_neighbourhood_search_response():
    ns = NeighbourhoodSearchResponse(
        query="affordable suburbs", results=[{"name": "oakwood", "score": 0.9}]
    )
    assert ns.query == "affordable suburbs"
    assert len(ns.results) == 1


@pytest.mark.parametrize("status", ["ok", "degraded", "error"])
def test_health_response_various_statuses(status):
    h = HealthResponse(status=status, service="property-sage", version="1.0.0")
    assert h.status == status
