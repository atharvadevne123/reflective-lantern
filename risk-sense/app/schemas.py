"""Pydantic schemas for Risk-Sense API request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskCategory = Literal["low", "medium", "high", "very_high"]


class CreditApplicationSchema(BaseModel):
    """Schema for validating credit application input.

    All numeric fields are range-validated on construction.
    """

    applicant_id: str = Field(
        ...,
        description="Unique applicant identifier",
        examples=["app-2026-001"],
    )
    debt_to_income: float = Field(
        ..., ge=0.0, le=10.0,
        description="Debt-to-income ratio (0 = no debt, 10 = extremely high)",
        examples=[0.35],
    )
    payment_history_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalised payment history score (0 = poor, 1 = perfect)",
        examples=[0.85],
    )
    credit_utilization: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of available credit currently used",
        examples=[0.30],
    )
    account_age_months: int = Field(
        ..., ge=1, le=600,
        description="Average age of open credit accounts in months",
        examples=[48],
    )
    num_open_accounts: int = Field(
        ..., ge=0, le=50,
        description="Number of open credit accounts",
        examples=[4],
    )
    num_derogatory_marks: int = Field(
        ..., ge=0, le=20,
        description="Number of derogatory marks (late payments, collections)",
        examples=[0],
    )
    annual_income: float = Field(
        ..., ge=0.0,
        description="Annual gross income in USD",
        examples=[75_000.0],
    )

    model_config = {"json_schema_extra": {"examples": [{
        "applicant_id": "app-2026-001",
        "debt_to_income": 0.35,
        "payment_history_score": 0.85,
        "credit_utilization": 0.30,
        "account_age_months": 48,
        "num_open_accounts": 4,
        "num_derogatory_marks": 0,
        "annual_income": 75000.0,
    }]}}


class CreditScoreSchema(BaseModel):
    """Schema for credit risk score response."""

    applicant_id: str = Field(..., description="Applicant identifier echoed from request")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite risk score (0=low, 1=high)")
    risk_category: RiskCategory = Field(..., description="Risk tier label")
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of loan default")
    recommendation: str = Field(..., description="Human-readable lending recommendation")


class HealthSchema(BaseModel):
    """Schema for health check response."""

    status: str
    model_loaded: bool
    version: str


class MetricsSchema(BaseModel):
    """Schema for aggregate metrics response."""

    total_predictions: int
    avg_default_probability: float
    high_risk_rate: float
