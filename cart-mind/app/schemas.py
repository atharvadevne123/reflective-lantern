"""Pydantic request and response schemas for Cart-Mind API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class UserItemFeatures(BaseModel):
    """Features for a single user-item interaction."""

    user_id: str = Field(..., description="Unique user identifier")
    item_id: str = Field(..., description="Unique item identifier")
    user_age: Annotated[int, Field(ge=0, le=120)] = 30
    purchase_count: Annotated[int, Field(ge=0)] = 0
    avg_order_value: Annotated[float, Field(ge=0)] = 0.0
    days_since_last_purchase: Annotated[int, Field(ge=0)] = 365
    days_since_registration: Annotated[int, Field(ge=1)] = 30
    session_count_7d: Annotated[int, Field(ge=0)] = 0
    cart_abandon_rate: Annotated[float, Field(ge=0, le=1)] = 0.0
    item_price: Annotated[float, Field(ge=0)] = 0.0
    item_avg_rating: Annotated[float, Field(ge=0, le=5)] = 0.0
    item_review_count: Annotated[int, Field(ge=0)] = 0
    item_inventory_level: Annotated[int, Field(ge=0)] = 0
    item_discount_pct: Annotated[float, Field(ge=0, le=100)] = 0.0
    view_count: Annotated[int, Field(ge=0)] = 0
    click_count: Annotated[int, Field(ge=0)] = 0
    wishlist_flag: Annotated[int, Field(ge=0, le=1)] = 0
    same_category_purchases: Annotated[int, Field(ge=0)] = 0


class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="User to generate recommendations for")
    user_age: Annotated[int, Field(ge=0, le=120)] = 30
    purchase_count: Annotated[int, Field(ge=0)] = 0
    avg_order_value: Annotated[float, Field(ge=0)] = 0.0
    days_since_last_purchase: Annotated[int, Field(ge=0)] = 365
    days_since_registration: Annotated[int, Field(ge=1)] = 30
    session_count_7d: Annotated[int, Field(ge=0)] = 0
    cart_abandon_rate: Annotated[float, Field(ge=0, le=1)] = 0.0
    top_k: Annotated[int, Field(ge=1, le=50)] = 10


class SimilarItemsRequest(BaseModel):
    item_id: str = Field(..., description="Seed item for similarity search")
    top_k: Annotated[int, Field(ge=1, le=50)] = 5
    item_price: Annotated[float, Field(ge=0)] = 50.0
    item_avg_rating: Annotated[float, Field(ge=0, le=5)] = 4.0
    item_review_count: Annotated[int, Field(ge=0)] = 100
    item_inventory_level: Annotated[int, Field(ge=0)] = 50
    item_discount_pct: Annotated[float, Field(ge=0, le=100)] = 0.0


class IntentResponse(BaseModel):
    user_id: str
    item_id: str
    purchase_probability: float
    will_purchase: bool
    confidence: str
    model_version: str
    correlation_id: str


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[dict]
    count: int
    correlation_id: str


class SimilarItemsResponse(BaseModel):
    seed_item_id: str
    similar_items: list[dict]
    count: int
    correlation_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    intent_requests: int
    recommend_requests: int
    similar_requests: int
    drift_alerts: int
    errors: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class DriftRequest(BaseModel):
    feature_values: dict[str, list[float]] = Field(
        ..., description="Map of feature name to list of recent values"
    )


class DriftResponse(BaseModel):
    results: dict[str, dict]
    drifted_features: list[str]
    total_checked: int


class BatchPredictRequest(BaseModel):
    """Batch of user-item pairs to score in a single call."""

    items: list[UserItemFeatures] = Field(
        ..., min_length=1, max_length=500, description="User-item pairs to score (1-500)"
    )


class BatchPredictResponse(BaseModel):
    predictions: list[dict]
    count: int
    mean_probability: float
    correlation_id: str


class ModelInfoResponse(BaseModel):
    model_version: str
    auc_mean: float | None
    auc_std: float | None
    n_features: int | None
    n_samples: int | None
    positive_rate: float | None
    ensemble_members: list[str]
    feature_stages: list[str]


class CacheStatsResponse(BaseModel):
    recommendation_cache: dict
    similarity_cache: dict
