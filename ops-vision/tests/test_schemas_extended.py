"""Extended tests for ops-vision app/schemas.py Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import MetricsPayload, RunbookSearchRequest

VALID_PAYLOAD = {
    "service_name": "api-svc",
    "cpu_usage_pct": 75.0,
    "memory_usage_pct": 60.0,
    "error_rate_per_min": 5.0,
    "latency_p99_ms": 200.0,
    "request_rate_per_sec": 100.0,
    "disk_io_util_pct": 30.0,
}


class TestMetricsPayloadValidation:
    def test_valid_payload_ok(self) -> None:
        payload = MetricsPayload(**VALID_PAYLOAD)
        assert payload.service_name == "api-svc"

    def test_cpu_above_100_raises(self) -> None:
        bad = {**VALID_PAYLOAD, "cpu_usage_pct": 101.0}
        with pytest.raises(ValidationError):
            MetricsPayload(**bad)

    def test_cpu_below_0_raises(self) -> None:
        bad = {**VALID_PAYLOAD, "cpu_usage_pct": -1.0}
        with pytest.raises(ValidationError):
            MetricsPayload(**bad)

    def test_empty_service_name_raises(self) -> None:
        bad = {**VALID_PAYLOAD, "service_name": ""}
        with pytest.raises(ValidationError):
            MetricsPayload(**bad)

    def test_service_name_stripped(self) -> None:
        p = MetricsPayload(**{**VALID_PAYLOAD, "service_name": "  my-svc  "})
        assert p.service_name == "my-svc"

    def test_negative_error_rate_raises(self) -> None:
        bad = {**VALID_PAYLOAD, "error_rate_per_min": -1.0}
        with pytest.raises(ValidationError):
            MetricsPayload(**bad)

    def test_memory_at_boundary_ok(self) -> None:
        p = MetricsPayload(**{**VALID_PAYLOAD, "memory_usage_pct": 100.0})
        assert p.memory_usage_pct == 100.0

    def test_zero_values_ok(self) -> None:
        p = MetricsPayload(**{
            **VALID_PAYLOAD,
            "cpu_usage_pct": 0.0,
            "memory_usage_pct": 0.0,
            "error_rate_per_min": 0.0,
            "latency_p99_ms": 0.0,
            "request_rate_per_sec": 0.0,
            "disk_io_util_pct": 0.0,
        })
        assert p.cpu_usage_pct == 0.0


class TestRunbookSearchRequest:
    def test_valid_request(self) -> None:
        req = RunbookSearchRequest(query="database connection failure", top_k=3)
        assert req.top_k == 3

    def test_query_too_short_raises(self) -> None:
        with pytest.raises(ValidationError):
            RunbookSearchRequest(query="ab", top_k=3)

    def test_top_k_above_10_raises(self) -> None:
        with pytest.raises(ValidationError):
            RunbookSearchRequest(query="valid query here", top_k=11)

    def test_top_k_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            RunbookSearchRequest(query="valid query here", top_k=0)

    def test_default_top_k(self) -> None:
        req = RunbookSearchRequest(query="some valid query string")
        assert req.top_k == 3
