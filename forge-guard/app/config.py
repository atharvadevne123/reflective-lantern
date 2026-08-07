"""Centralized settings management for Forge-Guard using Pydantic BaseSettings."""

from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    """Application-wide configuration loaded from environment variables.

    All values fall back to safe defaults so the service starts without
    any env vars configured (useful for local dev and test runs).
    """

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./forge_guard.db")

    # Model artefacts
    model_path: str = os.getenv("MODEL_PATH", "model.joblib")
    metrics_path: str = os.getenv("METRICS_PATH", "metrics.json")
    model_version: str = os.getenv("MODEL_VERSION", "1.0.0")

    # Rate limiting
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "60"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_json: bool = os.getenv("LOG_JSON", "false").lower() == "true"

    # CORS
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # Drift detection
    drift_window_hours: int = int(os.getenv("DRIFT_WINDOW_HOURS", "24"))
    drift_window_size: int = int(os.getenv("DRIFT_WINDOW_SIZE", "500"))
    drift_p_threshold: float = float(os.getenv("DRIFT_P_THRESHOLD", "0.05"))

    # AWS / S3
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "forge-guard/models")

    # FAISS anomaly detection
    faiss_anomaly_threshold: float = float(os.getenv("FAISS_ANOMALY_THRESHOLD", "50.0"))
    faiss_k_neighbours: int = int(os.getenv("FAISS_K_NEIGHBOURS", "5"))

    # Feature engineering
    lag_periods: int = int(os.getenv("LAG_PERIODS", "2"))
    rolling_window: int = int(os.getenv("ROLLING_WINDOW", "5"))

    # Connection pool tuning
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    db_pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))

    # Prediction confidence threshold
    prediction_confidence_threshold: float = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.5"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
