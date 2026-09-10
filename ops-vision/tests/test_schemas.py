"""Tests for Ops-Vision Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictRequest,
    ErrorResponse,
    MetricsPayload,
    ModelInfoResponse,
    RunbookSearchRequest,
    ServiceHealthStatus,
    SeverityLevel,
)


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

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("cpu_usage_pct", -1.0),
            ("cpu_usage_pct", 101.0),
            ("memory_usage_pct", -0.1),
            ("memory_usage_pct", 100.1),
            ("disk_io_util_pct", -1.0),
            ("disk_io_util_pct", 101.0),
            ("error_rate_per_min", -0.1),
            ("latency_p99_ms", -1.0),
            ("request_rate_per_sec", -1.0),
        ],
    )
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


class TestSeverityLevel:
    """Tests for the SeverityLevel enum."""

    @pytest.mark.parametrize("val", ["low", "medium", "high", "critical"])
    def test_valid_severity_levels(self, val):
        """All four severity strings are valid enum members."""
        level = SeverityLevel(val)
        assert level.value == val

    def test_invalid_severity_raises(self):
        """An unknown severity string raises ValueError."""
        with pytest.raises(ValueError):
            SeverityLevel("extreme")


class TestErrorResponse:
    """Tests for the ErrorResponse schema."""

    def test_minimal_error_response(self):
        """ErrorResponse with only detail parses correctly."""
        resp = ErrorResponse(detail="Something went wrong")
        assert resp.detail == "Something went wrong"
        assert resp.error_code is None

    def test_full_error_response(self):
        """ErrorResponse with all fields parses correctly."""
        resp = ErrorResponse(detail="Not found", error_code="E404", request_id="abc-123")
        assert resp.error_code == "E404"
        assert resp.request_id == "abc-123"


class TestBatchPredictRequest:
    """Tests for the BatchPredictRequest schema."""

    def _item(self) -> dict:
        return {
            "service_name": "svc",
            "cpu_usage_pct": 50.0,
            "memory_usage_pct": 50.0,
            "error_rate_per_min": 5.0,
            "latency_p99_ms": 200.0,
            "request_rate_per_sec": 100.0,
            "disk_io_util_pct": 30.0,
        }

    def test_single_item_accepted(self):
        """A batch of 1 item is valid."""
        req = BatchPredictRequest(items=[self._item()])
        assert len(req.items) == 1

    def test_empty_items_rejected(self):
        """An empty items list raises ValidationError."""
        with pytest.raises(ValidationError):
            BatchPredictRequest(items=[])

    def test_over_limit_rejected(self):
        """More than 100 items raises ValidationError."""
        with pytest.raises(ValidationError):
            BatchPredictRequest(items=[self._item()] * 101)


class TestServiceHealthStatus:
    """Tests for the ServiceHealthStatus schema."""

    def test_valid_health_status_parses(self):
        """Valid ServiceHealthStatus is accepted."""
        status = ServiceHealthStatus(
            service_name="payments-api",
            total_predictions=100,
            incident_count=10,
            incident_rate=0.1,
            avg_confidence=0.75,
        )
        assert status.service_name == "payments-api"

    def test_incident_rate_out_of_range_rejected(self):
        """incident_rate > 1 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ServiceHealthStatus(
                service_name="svc",
                total_predictions=10,
                incident_count=5,
                incident_rate=1.5,
                avg_confidence=0.5,
            )


class TestModelInfoResponse:
    """Tests for the ModelInfoResponse schema."""

    def test_minimal_model_info(self):
        """ModelInfoResponse with only required fields parses correctly."""
        resp = ModelInfoResponse(model_version="1.0.0", model_loaded=True)
        assert resp.model_version == "1.0.0"
        assert resp.model_loaded is True
        assert resp.estimators is None

    def test_full_model_info(self):
        """ModelInfoResponse with estimators list parses correctly."""
        resp = ModelInfoResponse(
            model_version="1.0.0",
            model_loaded=True,
            estimators=["xgb", "lgbm", "rf"],
        )
        assert len(resp.estimators) == 3
