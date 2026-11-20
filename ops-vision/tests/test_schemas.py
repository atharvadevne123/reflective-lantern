"""Tests for Ops-Vision Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from app.schemas import MetricsPayload, RunbookSearchRequest


class TestMetricsPayload:
    """Tests for the MetricsPayload input schema."""

    def _valid(self) -> dict:
        return {
            "service_name": "payments-api",
            "cpu_usage_pct": 85.0,
            "memory_usage_pct": 88.0,
            "error_rate_per_min": 62.0,
            "latency_p99_ms": 1450.0,
            "request_rate_per_sec": 45.0,
            "disk_io_util_pct": 80.0,
        }

    def test_valid_payload_parses(self):
        """Valid payload is accepted without error."""
        payload = MetricsPayload(**self._valid())
        assert payload.service_name == "payments-api"

    def test_service_name_stripped(self):
        """Whitespace is stripped from service_name."""
        data = self._valid()
        data["service_name"] = "  my-service  "
        payload = MetricsPayload(**data)
        assert payload.service_name == "my-service"

    @pytest.mark.parametrize("field,bad_value", [
        ("cpu_usage_pct", -1.0),
        ("cpu_usage_pct", 101.0),
        ("memory_usage_pct", -0.1),
        ("memory_usage_pct", 100.1),
        ("disk_io_util_pct", -1.0),
        ("disk_io_util_pct", 101.0),
        ("error_rate_per_min", -0.1),
        ("latency_p99_ms", -1.0),
        ("request_rate_per_sec", -1.0),
    ])
    def test_out_of_range_values_rejected(self, field, bad_value):
        """Values outside the allowed range raise ValidationError."""
        data = self._valid()
        data[field] = bad_value
        with pytest.raises(ValidationError):
            MetricsPayload(**data)

    def test_empty_service_name_rejected(self):
        """Empty service_name raises ValidationError."""
        data = self._valid()
        data["service_name"] = ""
        with pytest.raises(ValidationError):
            MetricsPayload(**data)

    def test_missing_field_raises(self):
        """Missing required field raises ValidationError."""
        data = self._valid()
        del data["cpu_usage_pct"]
        with pytest.raises(ValidationError):
            MetricsPayload(**data)

    @pytest.mark.parametrize("cpu", [0.0, 50.0, 100.0])
    def test_valid_cpu_boundary_values(self, cpu):
        """CPU at the boundaries 0 and 100 are valid."""
        data = self._valid()
        data["cpu_usage_pct"] = cpu
        payload = MetricsPayload(**data)
        assert payload.cpu_usage_pct == cpu


class TestRunbookSearchRequest:
    """Tests for the RunbookSearchRequest schema."""

    def test_valid_request_parses(self):
        """Valid search request is accepted."""
        req = RunbookSearchRequest(query="high cpu usage", top_k=3)
        assert req.top_k == 3

    def test_query_too_short_rejected(self):
        """Query shorter than 3 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            RunbookSearchRequest(query="ab", top_k=3)

    def test_top_k_zero_rejected(self):
        """top_k=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            RunbookSearchRequest(query="valid query", top_k=0)

    def test_top_k_too_large_rejected(self):
        """top_k > 10 raises ValidationError."""
        with pytest.raises(ValidationError):
            RunbookSearchRequest(query="valid query", top_k=11)

    def test_default_top_k(self):
        """Default top_k is 3."""
        req = RunbookSearchRequest(query="network issue")
        assert req.top_k == 3

    @pytest.mark.parametrize("top_k", [1, 5, 10])
    def test_valid_top_k_values(self, top_k):
        """top_k values 1–10 are all valid."""
        req = RunbookSearchRequest(query="disk io saturation", top_k=top_k)
        assert req.top_k == top_k
