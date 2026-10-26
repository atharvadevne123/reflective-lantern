"""Application configuration for Ticket-Oracle.

Centralises all environment-variable-backed settings using
pydantic-settings so values can be overridden in CI or production
without touching source files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model storage
    model_dir: Path = Field(default=Path("models"), alias="MODEL_DIR")
    model_dir_env: str = Field(default="models", alias="MODEL_DIR")

    # API behaviour
    api_rate_limit: str = Field(default="60/minute", alias="API_RATE_LIMIT")
    max_batch_size: int = Field(default=100, alias="MAX_BATCH_SIZE")
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")

    # Cache
    cache_maxsize: int = Field(default=512, alias="CACHE_MAXSIZE")
    cache_ttl_seconds: float = Field(default=300.0, alias="CACHE_TTL_SECONDS")

    # Database
    database_url: str = Field(default="sqlite:///./ticket_oracle.db", alias="DATABASE_URL")

    # Monitoring
    drift_window_hours: int = Field(default=24, alias="DRIFT_WINDOW_HOURS")
    spike_z_threshold: float = Field(default=2.5, alias="SPIKE_Z_THRESHOLD")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "populate_by_name": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
