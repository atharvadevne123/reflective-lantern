"""Middleware tests for Risk-Sense."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.middleware import CorrelationIdMiddleware, RateLimitMiddleware


class TestCorrelationIdMiddleware:
    def test_response_has_correlation_id_header(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert "X-Correlation-ID" in response.headers

    def test_client_supplied_correlation_id_is_echoed(self, client: TestClient) -> None:
        custom_id = "my-custom-request-id-123"
        response = client.get("/api/v1/health", headers={"X-Correlation-ID": custom_id})
        assert response.headers.get("X-Correlation-ID") == custom_id

    def test_correlation_id_is_string(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        correlation_id = response.headers.get("X-Correlation-ID")
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0


class TestRateLimitMiddleware:
    def test_normal_requests_succeed(self, client: TestClient) -> None:
        for _ in range(5):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
