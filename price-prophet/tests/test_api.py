"""Tests for app/api routes and schemas."""

from __future__ import annotations


def test_health_endpoint():
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_price_endpoint_valid():
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    payload = {
        "product_id": "P1",
        "base_price": 100.0,
        "demand": 50.0,
    }
    response = client.post("/api/v1/price", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_price" in data
    assert data["optimal_price"] > 0


def test_price_schema_product_id():
    from app.api.schemas import PriceRequest

    req = PriceRequest(product_id="X1", base_price=50.0, demand=100.0)
    assert req.product_id == "X1"


def test_price_response_schema():
    from app.api.schemas import PriceResponse

    resp = PriceResponse(
        product_id="X1",
        optimal_price=55.0,
        revenue_estimate=5500.0,
        strategy="dynamic",
        confidence=0.8,
    )
    assert resp.confidence == 0.8
