"""Configuration management for Risk-Sense using environment variables."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


class Settings:
    """Application settings loaded from environment variables.

    All settings have sensible defaults for local development.
    """

    database_url: str
    model_path: str
    pipeline_path: str
    api_host: str
    api_port: int
    log_level: str
    drift_ks_threshold: float
    drift_psi_threshold: float
    min_samples_for_drift: int
    retrain_interval_hours: int
    min_records_for_retrain: int

    def __init__(self) -> None:
        """Load all settings from environment variables."""
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./risk_sense.db")
        self.model_path = os.getenv("MODEL_PATH", "/tmp/risk_sense_model.pkl")
        self.pipeline_path = os.getenv("PIPELINE_PATH", "/tmp/risk_sense_pipeline.pkl")
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.drift_ks_threshold = float(os.getenv("DRIFT_KS_THRESHOLD", "0.05"))
        self.drift_psi_threshold = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.25"))
        self.min_samples_for_drift = int(os.getenv("MIN_SAMPLES_FOR_DRIFT", "10"))
        self.retrain_interval_hours = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))
        self.min_records_for_retrain = int(os.getenv("MIN_RECORDS_FOR_RETRAIN", "500"))

    def as_dict(self) -> dict[str, Any]:
        """Return settings as a dict (excluding sensitive values).

        Returns:
            Dict of non-sensitive settings.
        """
        return {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "drift_ks_threshold": self.drift_ks_threshold,
            "drift_psi_threshold": self.drift_psi_threshold,
            "min_samples_for_drift": self.min_samples_for_drift,
            "retrain_interval_hours": self.retrain_interval_hours,
            "min_records_for_retrain": self.min_records_for_retrain,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton.

    Returns:
        Loaded Settings instance.
    """
    settings = Settings()
    logger.info("Settings loaded: port=%d log_level=%s", settings.api_port, settings.log_level)
    return settings
