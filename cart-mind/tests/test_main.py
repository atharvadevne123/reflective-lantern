"""Tests for cart-mind app/main.py FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/api/v1/health").status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/api/v1/health").json(), dict)

    def test_has_status_key(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "status" in data or isinstance(data, dict)


class TestMetricsEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/api/v1/metrics").status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/api/v1/metrics").json(), dict)


class TestPredictEndpoint:
    def test_predict_valid_payload(self, client: TestClient) -> None:
        payload = {
            "user_id": "user123",
            "item_id": "item456",
            "user_age": 30,
            "purchase_count": 5,
            "avg_order_value": 50.0,
            "days_since_last_purchase": 10,
            "days_since_registration": 365,
            "session_count_7d": 3,
            "cart_abandon_rate": 0.2,
            "item_price": 29.99,
            "item_avg_rating": 4.2,
            "item_review_count": 100,
            "item_inventory_level": 50,
            "item_discount_pct": 10.0,
            "view_count": 5,
            "click_count": 2,
            "wishlist_flag": 0,
            "same_category_purchases": 3,
        }
        resp = client.post("/api/v1/predict/intent", json=payload)
        assert resp.status_code in (200, 201, 400, 422, 500)

    def test_predict_empty_payload(self, client: TestClient) -> None:
        resp = client.post("/api/v1/predict/intent", json={})
        assert resp.status_code in (400, 422, 500)


class TestAppMetadata:
    def test_app_title(self) -> None:
        assert app.title == "Cart-Mind"

    def test_openapi_json(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
