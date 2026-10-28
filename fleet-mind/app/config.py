"""Configuration settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""

    # Database
    database_url: str = Field(
        default="sqlite:///./fleet_mind.db",
        description="SQLAlchemy database URL",
    )

    # Model
    model_path: str = Field(
        default="/tmp/fleet_mind_model.joblib",
        description="Path where the trained model is stored",
    )
    metrics_path: str = Field(
        default="/tmp/fleet_mind_metrics.json",
        description="Path where model metrics are stored",
    )

    # Drift monitoring
    drift_threshold: float = Field(
        default=0.05,
        description="KS-test p-value threshold below which drift is flagged",
    )
    drift_window: int = Field(
        default=500,
        description="Number of recent predictions used for drift comparison",
    )

    # API
    api_title: str = Field(default="Fleet-Mind", description="OpenAPI title")
    api_version: str = Field(default="1.0.0", description="API version")
    log_level: str = Field(default="INFO", description="Logging level")
    log_json: bool = Field(default=False, description="Emit JSON-formatted logs")

    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
