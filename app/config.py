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
    min_reference_samples: int = field(
        default_factory=lambda: int(os.getenv("MIN_REFERENCE_SAMPLES", "10"))
    )
    cors_origins: str = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*"))
    enable_json_logs: bool = field(
        default_factory=lambda: os.getenv("ENABLE_JSON_LOGS", "false").lower() == "true"
    )

    @property
    def is_postgres(self) -> bool:
        """True when the configured database is PostgreSQL."""
        return self.database_url.startswith("postgresql")

    @property
    def is_debug(self) -> bool:
        """True when LOG_LEVEL is set to DEBUG."""
        return self.log_level.upper() == "DEBUG"

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a list split on commas."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    """Return a freshly resolved Settings instance."""
    return Settings()
