"""Centralised application settings loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Runtime configuration resolved from environment variables."""

    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./logistics_flow.db")
    )
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "model.joblib"))
    feature_pipeline_path: str = field(
        default_factory=lambda: os.getenv("FEATURE_PIPELINE_PATH", "feature_pipeline.joblib")
    )
    metrics_path: str = field(default_factory=lambda: os.getenv("METRICS_PATH", "metrics.json"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    model_version: str = field(default_factory=lambda: os.getenv("MODEL_VERSION", "1.0.0"))
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    )
    drift_window: int = field(default_factory=lambda: int(os.getenv("DRIFT_WINDOW", "100")))
    secret_key: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY", "change-me-in-production")
    )
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "4")))
    request_timeout_s: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_S", "30"))
    )
    rate_limit_requests: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    )
    rate_limit_window_s: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))
    )
    drift_ks_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_KS_THRESHOLD", "0.05"))
    )
    reference_buffer_size: int = field(
        default_factory=lambda: int(os.getenv("REFERENCE_BUFFER_SIZE", "500"))
    )

    @property
    def is_postgres(self) -> bool:
        """True when the configured database is PostgreSQL."""
        return self.database_url.startswith("postgresql")


def get_settings() -> Settings:
    """Return a freshly resolved Settings instance."""
    return Settings()


__all__ = ["Settings", "get_settings"]
