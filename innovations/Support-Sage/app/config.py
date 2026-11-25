"""Application configuration loaded from environment."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    log_level: str
    drift_window_days: int
    drift_p_threshold: float
    model_path: str
    encoder_path: str
    regressor_path: str

def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./support_sage.db"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        drift_window_days=int(os.getenv("DRIFT_WINDOW_DAYS", "7")),
        drift_p_threshold=float(os.getenv("DRIFT_P_VALUE_THRESHOLD", "0.05")),
        model_path=os.getenv("MODEL_PATH", "models/priority_model.joblib"),
        encoder_path=os.getenv("ENCODER_PATH", "models/priority_encoder.joblib"),
        regressor_path=os.getenv("REGRESSOR_PATH", "models/resolution_model.joblib"),
    )
