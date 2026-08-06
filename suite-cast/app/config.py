"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Central settings object populated from environment.

    Attributes:
        database_url: SQLAlchemy connection string (SQLite dev, PostgreSQL prod).
        model_version: Version tag stamped on every logged prediction.
        base_room_rate: Baseline nightly rate the pricing multiplier applies to.
        rate_limit_per_minute: Per-IP sliding-window request budget.
        api_prefix: URL prefix for versioned API routes.
        log_level: Root logging level name (DEBUG/INFO/WARNING/ERROR).
        environment: Deployment environment label (development/production).
    """

    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./suite_cast.db")
    )
    model_version: str = field(default_factory=lambda: os.getenv("MODEL_VERSION", "1.0.0"))
    base_room_rate: float = field(
        default_factory=lambda: float(os.getenv("BASE_ROOM_RATE", "150.0"))
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    )
    api_prefix: str = "/api/v1"
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))


settings = Settings()
