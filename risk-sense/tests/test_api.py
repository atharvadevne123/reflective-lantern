"""API endpoint tests for Risk-Sense."""

from __future__ import annotations


class TestHealthEndpoint:
    def test_health_returns_200(self, client) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client) -> None:
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_is_healthy(self, client) -> None:
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"


class TestVersionEndpoint:
    def test_version_returns_200(self, client) -> None:
        response = client.get("/api/v1/version")
        assert response.status_code == 200

    def test_version_has_version_key(self, client) -> None:
        data = client.get("/api/v1/version").json()
        assert "version" in data
        assert data["version"] == "1.0.0"


class TestScoreEndpoint:
    def test_score_valid_application_returns_200(self, client, sample_application) -> None:
        response = client.post("/api/v1/score", json=sample_application)
        assert response.status_code == 200

    def test_score_response_contains_all_fields(self, client, sample_application) -> None:
        data = client.post("/api/v1/score", json=sample_application).json()
        assert "applicant_id" in data
        assert "risk_score" in data
        assert "risk_category" in data
        assert "default_probability" in data
        assert "recommendation" in data

    def test_score_probability_is_between_0_and_1(self, client, sample_application) -> None:
        data = client.post("/api/v1/score", json=sample_application).json()
        assert 0.0 <= data["default_probability"] <= 1.0

    def test_score_risk_category_is_valid(self, client, sample_application) -> None:
        data = client.post("/api/v1/score", json=sample_application).json()
        assert data["risk_category"] in ("low", "medium", "high", "very_high")

    def test_score_high_risk_returns_high_category(self, client, high_risk_application) -> None:
        data = client.post("/api/v1/score", json=high_risk_application).json()
        assert data["risk_category"] in ("high", "very_high")

    def test_score_missing_required_field_returns_422(self, client) -> None:
        incomplete = {"debt_to_income": 0.3, "payment_history_score": 0.8}
        response = client.post("/api/v1/score", json=incomplete)
        assert response.status_code == 422

    def test_score_invalid_dti_range_returns_422(self, client, sample_application) -> None:
        bad = {**sample_application, "debt_to_income": -1.0}
        response = client.post("/api/v1/score", json=bad)
        assert response.status_code == 422

    def test_score_credit_utilization_over_1_returns_422(self, client, sample_application) -> None:
        bad = {**sample_application, "credit_utilization": 1.5}
        response = client.post("/api/v1/score", json=bad)
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client) -> None:
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

    def test_metrics_response_structure(self, client) -> None:
        data = client.get("/api/v1/metrics").json()
        assert "total_predictions" in data
        assert "avg_default_probability" in data
        assert "high_risk_rate" in data

    def test_metrics_total_non_negative(self, client) -> None:
        data = client.get("/api/v1/metrics").json()
        assert data["total_predictions"] >= 0


class TestDriftEndpoint:
    def test_drift_returns_200(self, client) -> None:
        response = client.get("/api/v1/drift")
        assert response.status_code == 200

    def test_drift_insufficient_data_message(self, client) -> None:
        data = client.get("/api/v1/drift").json()
        # Fresh DB won't have 20+ records
        assert "status" in data or "drift_analysis" in data
