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


def test_version_endpoint() -> None:
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data


def test_metrics_endpoint() -> None:
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_price_endpoint_negative_price_rejected() -> None:
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/price", json={"product_id": "PROD-001", "base_price": -10.0, "demand": 50.0}
    )
    assert response.status_code == 422


def test_price_endpoint_zero_demand() -> None:
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/price", json={"product_id": "PROD-002", "base_price": 100.0, "demand": 0.0}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["revenue_estimate"] == 0.0


def test_batch_price_endpoint() -> None:
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    payload = {
        "items": [
            {"product_id": "P1", "base_price": 100.0, "demand": 50.0},
            {"product_id": "P2", "base_price": 200.0, "demand": 30.0},
        ]
    }
    response = client.post("/api/v1/price/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_batch_price_endpoint_empty_items_rejected() -> None:
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    response = client.post("/api/v1/price/batch", json={"items": []})
    assert response.status_code == 422
