"""Tests for environment-driven settings."""

from __future__ import annotations

import dataclasses

import pytest

from app.config import Settings, get_settings


def test_defaults_present(monkeypatch):
    for var in ("DATABASE_URL", "MODEL_PATH", "LOG_LEVEL", "RATE_LIMIT_PER_MINUTE"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.database_url.startswith("sqlite")
    assert s.log_level == "INFO"
    assert s.rate_limit_per_minute == 120


def test_env_override(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")
    s = Settings()
    assert s.log_level == "DEBUG"
    assert s.rate_limit_per_minute == 5


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@h/db", True),
        ("sqlite:///./x.db", False),
    ],
)
def test_is_postgres(monkeypatch, url, expected):
    monkeypatch.setenv("DATABASE_URL", url)
    assert Settings().is_postgres is expected


def test_get_settings_returns_settings():
    assert isinstance(get_settings(), Settings)


def test_settings_frozen():
    s = Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.log_level = "TRACE"  # type: ignore[misc]


def test_model_path_default(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    assert Settings().model_path == "model.joblib"


def test_model_path_override(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/tmp/custom.joblib")
    assert Settings().model_path == "/tmp/custom.joblib"


def test_metrics_path_default(monkeypatch):
    monkeypatch.delenv("METRICS_PATH", raising=False)
    assert Settings().metrics_path == "metrics.json"


def test_drift_window_default(monkeypatch):
    monkeypatch.delenv("DRIFT_WINDOW", raising=False)
    assert Settings().drift_window == 100


def test_drift_window_override(monkeypatch):
    monkeypatch.setenv("DRIFT_WINDOW", "250")
    assert Settings().drift_window == 250


def test_model_version_default(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    assert Settings().model_version == "1.0.0"


def test_new_settings_defaults(monkeypatch):
    for var in (
        "SECRET_KEY",
        "MAX_WORKERS",
        "REQUEST_TIMEOUT_S",
        "RATE_LIMIT_REQUESTS",
        "RATE_LIMIT_WINDOW_S",
        "DRIFT_KS_THRESHOLD",
        "REFERENCE_BUFFER_SIZE",
    ):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.secret_key == "change-me-in-production"
    assert s.max_workers == 4
    assert s.request_timeout_s == 30
    assert s.rate_limit_requests == 100
    assert s.rate_limit_window_s == 60
    assert s.drift_ks_threshold == 0.05
    assert s.reference_buffer_size == 500


def test_new_settings_overrides(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "super-secret")
    monkeypatch.setenv("MAX_WORKERS", "8")
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "60")
    monkeypatch.setenv("DRIFT_KS_THRESHOLD", "0.1")
    monkeypatch.setenv("REFERENCE_BUFFER_SIZE", "1000")
    s = Settings()
    assert s.secret_key == "super-secret"
    assert s.max_workers == 8
    assert s.request_timeout_s == 60
    assert s.drift_ks_threshold == 0.1
    assert s.reference_buffer_size == 1000
