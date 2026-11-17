"""Tests for Property-Sage API schemas."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import (
    DriftCheckResponse,
    HealthResponse,
    MetricsResponse,
    NeighbourhoodReportResponse,
    PredictionResponse,
)


def test_health_response_fields():
    h = HealthResponse(status="ok", service="property-sage", version="1.0.0")
    assert h.status == "ok"
    assert h.service == "property-sage"
    assert h.version == "1.0.0"


def test_prediction_response_fields():
    pr = PredictionResponse(
        request_id="abc123",
        predicted_price=450000.0,
        predicted_rental_yield=0.05,
        estimated_annual_rental=22500.0,
        estimated_monthly_rental=1875.0,
        neighborhood="suburb",
        property_type="house",
    )
    assert pr.request_id == "abc123"
    assert pr.predicted_price == 450000.0


def test_drift_check_response_defaults():
    dr = DriftCheckResponse(status="ok")
    assert dr.drift_detected is None
    assert dr.checks == []


def test_drift_check_response_with_drift():
    dr = DriftCheckResponse(status="drifted", drift_detected=True, checks=[{"feature": "sqft"}])
    assert dr.drift_detected is True
    assert len(dr.checks) == 1


def test_metrics_response_defaults():
    mr = MetricsResponse()
    assert isinstance(mr.model_performance, dict)
    assert isinstance(mr.prediction_stats, dict)


def test_neighbourhood_report_response():
    nr = NeighbourhoodReportResponse(neighbourhood="downtown", market_report="Strong growth")
    assert nr.neighbourhood == "downtown"
    assert "growth" in nr.market_report


@pytest.mark.parametrize("status", ["ok", "degraded", "unhealthy"])
def test_health_response_various_statuses(status):
    h = HealthResponse(status=status, service="ps", version="1.0.0")
    assert h.status == status
