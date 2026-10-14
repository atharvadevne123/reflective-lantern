"""Centralised application configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration snapshot.

    Attributes:
        database_url: SQLAlchemy-compatible DB connection string.
        model_path: Path to the persisted model artifact.
        metrics_path: Path to the persisted training-metrics JSON.
        log_level: Logging level string (DEBUG/INFO/WARNING/ERROR).
        api_rate_limit: Max requests per minute per IP.
        cache_ttl_seconds: Prediction cache TTL in seconds.
        cache_maxsize: Maximum number of cached predictions.
        drift_threshold: KS-test p-value below which drift is flagged.
        horizon_days: Default prediction horizon in trading days.
        cors_origins: Allowed CORS origins.
    """

    database_url: str
    model_path: str
    metrics_path: str
    log_level: str
    api_rate_limit: int
    cache_ttl_seconds: float
    cache_maxsize: int
    drift_threshold: float
    horizon_days: int
    cors_origins: list[str]


def load_config() -> AppConfig:
    """Build AppConfig from environment variables with sensible defaults.

    Returns:
        Populated AppConfig instance.
    """
    cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    return AppConfig(
        database_url=os.getenv("DATABASE_URL", "sqlite:///trade_sage.db"),
        model_path=os.getenv("MODEL_PATH", "trade_sage_model.pkl"),
        metrics_path=os.getenv("METRICS_PATH", "trade_sage_metrics.json"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_rate_limit=int(os.getenv("API_RATE_LIMIT", "200")),
        cache_ttl_seconds=float(os.getenv("CACHE_TTL_SECONDS", "300")),
        cache_maxsize=int(os.getenv("CACHE_MAXSIZE", "512")),
        drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "0.05")),
        horizon_days=int(os.getenv("HORIZON_DAYS", "5")),
        cors_origins=cors_origins,
    )


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the singleton AppConfig, loading it on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
