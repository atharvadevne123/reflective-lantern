"""Pydantic v2 request/response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ClaimRequest(BaseModel):
    claim_id: str = Field(..., description="Unique claim identifier")
    claim_age_days: float = Field(0.0, ge=0, le=36500, description="Days since incident to report date")
    policy_limit: float = Field(100_000.0, gt=0, le=10_000_000, description="Policy coverage limit in USD")
    claimed_amount: float = Field(10_000.0, ge=0, description="Amount claimed in USD")
    num_prior_claims: float = Field(0.0, ge=0, le=100, description="Number of prior claims on policy")
    policy_tenure_years: float = Field(1.0, gt=0, le=50, description="Years the policy has been active")
    geo_risk_tier: float = Field(2.5, ge=1.0, le=5.0, description="Geographic risk tier (1=low, 5=high)")
    days_since_last_claim: float = Field(365.0, ge=0, description="Days since last prior claim")
    policy_type: str = Field("auto", description="Policy type: home, auto, life, commercial")

    @field_validator("policy_type")
    @classmethod
    def validate_policy_type(cls, v: str) -> str:
        allowed = {"home", "auto", "life", "commercial"}
        if v.lower() not in allowed:
            raise ValueError(f"policy_type must be one of {allowed}")
        return v.lower()

    @field_validator("claimed_amount")
    @classmethod
    def claimed_le_limit(cls, v: float, info: Any) -> float:
        limit = info.data.get("policy_limit", 10_000_000)
        if v > limit * 1.5:
            raise ValueError("claimed_amount cannot exceed 1.5x policy_limit")
        return v


class ClaimResponse(BaseModel):
    claim_id: str
    severity_class: int
    severity_label: str
    severity_score: float
    probabilities: dict[str, float]
    anomaly_flag: int
    triage_priority: int
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    f1_macro_mean: float | None
    f1_macro_std: float | None
    n_features: int | None
    n_samples: int | None
    severity_classes: list[str]
    total_predictions: int
    drift_alerts: int


class DriftResponse(BaseModel):
    feature: str
    ks_statistic: float | None
    p_value: float | None
    drift_detected: bool
    reference_size: int
    current_size: int


class PremiumRequest(BaseModel):
    base_premium: float = Field(..., gt=0, description="Current annual premium in USD")
    triage_priority: int = Field(..., ge=1, le=4, description="Claim triage priority 1–4")
    severity_label: str = Field(..., description="Predicted claim severity label")
    claim_frequency: float = Field(0.0, ge=0, description="Claims per year on this policy")


class PremiumResponse(BaseModel):
    base_premium: float
    risk_multiplier: float
    adjusted_premium: float
    recommendation: str
