"""Application configuration loaded from environment variables."""

from functools import lru_cache

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
