"""Schema validation tests for Risk-Sense."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import CreditApplicationSchema, CreditScoreSchema, HealthSchema, MetricsSchema


class TestCreditApplicationSchema:
    def test_valid_application_constructs(self) -> None:
        app = CreditApplicationSchema(
            applicant_id="test-001",
            debt_to_income=0.35,
            payment_history_score=0.85,
            credit_utilization=0.30,
            account_age_months=48,
            num_open_accounts=4,
            num_derogatory_marks=0,
            annual_income=75_000.0,
        )
        assert app.applicant_id == "test-001"
        assert app.debt_to_income == 0.35

    def test_dti_above_10_raises(self) -> None:
        with pytest.raises(ValidationError):
            CreditApplicationSchema(
                applicant_id="x", debt_to_income=11.0, payment_history_score=0.5,
                credit_utilization=0.5, account_age_months=12, num_open_accounts=2,
                num_derogatory_marks=0, annual_income=50_000.0
            )

    def test_negative_income_raises(self) -> None:
        with pytest.raises(ValidationError):
            CreditApplicationSchema(
                applicant_id="x", debt_to_income=0.3, payment_history_score=0.5,
                credit_utilization=0.5, account_age_months=12, num_open_accounts=2,
                num_derogatory_marks=0, annual_income=-100.0
            )

    def test_utilization_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            CreditApplicationSchema(
                applicant_id="x", debt_to_income=0.3, payment_history_score=0.5,
                credit_utilization=1.5, account_age_months=12, num_open_accounts=2,
                num_derogatory_marks=0, annual_income=50_000.0
            )

    @pytest.mark.parametrize("dti", [0.0, 0.5, 5.0, 10.0])
    def test_valid_dti_range(self, dti: float) -> None:
        app = CreditApplicationSchema(
            applicant_id="x", debt_to_income=dti, payment_history_score=0.5,
            credit_utilization=0.5, account_age_months=12, num_open_accounts=2,
            num_derogatory_marks=0, annual_income=50_000.0
        )
        assert app.debt_to_income == dti


class TestCreditScoreSchema:
    def test_valid_response_constructs(self) -> None:
        resp = CreditScoreSchema(
            applicant_id="test-001",
            risk_score=0.25,
            risk_category="medium",
            default_probability=0.25,
            recommendation="Approve with standard terms",
        )
        assert resp.risk_category == "medium"

    def test_invalid_risk_category_raises(self) -> None:
        with pytest.raises(ValidationError):
            CreditScoreSchema(
                applicant_id="x", risk_score=0.5, risk_category="unknown",
                default_probability=0.5, recommendation="test"
            )


class TestHealthSchema:
    def test_health_schema_constructs(self) -> None:
        h = HealthSchema(status="healthy", model_loaded=True, version="1.0.0")
        assert h.status == "healthy"
        assert h.model_loaded is True


class TestMetricsSchema:
    def test_metrics_schema_constructs(self) -> None:
        m = MetricsSchema(total_predictions=100, avg_default_probability=0.3, high_risk_rate=0.15)
        assert m.total_predictions == 100
