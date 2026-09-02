"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central settings object for Ops-Vision.

    All values can be overridden via environment variables or a .env file.
    """

    app_name: str = "Ops-Vision"
    app_version: str = "1.0.0"
    debug: bool = False

    database_url: str = "postgresql://ops:ops@localhost:5432/opsvision"
    model_path: str = "/tmp/ops_vision_model.pkl"
    feature_pipeline_path: str = "/tmp/ops_vision_pipeline.pkl"

    faiss_index_path: str = "/tmp/ops_vision_faiss.index"
    runbooks_path: str = "data/runbooks/sample_runbooks.json"
    embedding_dim: int = 64

    reference_window_size: int = 1000
    current_window_size: int = 200
    drift_threshold: float = 0.05

    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("drift_threshold")
    @classmethod
    def validate_drift_threshold(cls, v: float) -> float:
        """Drift threshold must be strictly between 0 and 1."""
        if not 0.0 < v < 1.0:
            raise ValueError(f"drift_threshold must be in (0, 1), got {v}")
        return v

    @field_validator("rate_limit_requests")
    @classmethod
    def validate_rate_limit_requests(cls, v: int) -> int:
        """Rate limit must be a positive integer."""
        if v <= 0:
            raise ValueError(f"rate_limit_requests must be > 0, got {v}")
        return v

    @field_validator("rate_limit_window_seconds")
    @classmethod
    def validate_rate_limit_window(cls, v: int) -> int:
        """Window must be a positive number of seconds."""
        if v <= 0:
            raise ValueError(f"rate_limit_window_seconds must be > 0, got {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
