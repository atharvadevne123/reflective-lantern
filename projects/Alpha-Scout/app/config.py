"""Application configuration via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Alpha-Scout runtime configuration loaded from environment variables."""

    database_url: str = Field("sqlite:///./alpha_scout.db", alias="DATABASE_URL")
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    model_version: str = Field("1.0.0", alias="MODEL_VERSION")
    model_path: str = Field("model.joblib", alias="MODEL_PATH")
    drift_p_value_threshold: float = Field(0.05, alias="DRIFT_P_VALUE_THRESHOLD")
    max_prediction_log: int = Field(10_000, alias="MAX_PREDICTION_LOG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    rate_limit_requests: int = Field(60, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: float = Field(60.0, alias="RATE_LIMIT_WINDOW")

    class Config:
        env_file = ".env"
        populate_by_name = True


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
