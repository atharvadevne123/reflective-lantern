"""Shared Pydantic schemas and input validation helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PostFeaturesFull(BaseModel):
    """Extended post features with optional engagement history."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Raw post text.",
        examples=["Just launched! #innovation #tech @everyone 🚀"],
    )
    author_followers: int = Field(
        ..., ge=0, le=500_000_000, description="Author follower count."
    )
    has_media: bool = Field(False, description="Post includes image/video.")
    has_link: bool = Field(False, description="Post contains a URL.")
    hashtag_count: int = Field(0, ge=0, le=100)
    mention_count: int = Field(0, ge=0, le=100)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    author_avg_engagement: float = Field(0.0, ge=0.0)
    is_reply: bool = Field(False)
    is_quote: bool = Field(False)

    @field_validator("text")
    @classmethod
    def strip_and_validate_text(cls, v: str) -> str:
        """Strip whitespace and reject blank text."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be blank after stripping whitespace")
        return stripped

    @field_validator("author_avg_engagement")
    @classmethod
    def validate_engagement(cls, v: float) -> float:
        """Cap average engagement to a reasonable maximum."""
        if v > 10_000_000:
            raise ValueError("author_avg_engagement exceeds maximum (10M)")
        return v


class BatchPredictRequest(BaseModel):
    """Request body for bulk prediction of up to 50 posts."""

    posts: list[PostFeaturesFull] = Field(..., min_length=1, max_length=50)


class BatchPredictResponse(BaseModel):
    """Response body for bulk prediction."""

    results: list[dict[str, Any]]
    total: int
    model_version: str


class ErrorResponse(BaseModel):
    """Standard error response shape."""

    detail: str
    code: str = "INTERNAL_ERROR"
