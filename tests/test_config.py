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


def test_drift_window_default(monkeypatch):
    monkeypatch.delenv("DRIFT_WINDOW", raising=False)
    s = Settings()
    assert s.drift_window == 100


def test_model_version_default(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    s = Settings()
    assert s.model_version == "1.0.0"


def test_rate_limit_from_env(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "30")
    s = Settings()
    assert s.rate_limit_per_minute == 30


def test_feature_pipeline_path_from_env(monkeypatch):
    monkeypatch.setenv("FEATURE_PIPELINE_PATH", "/tmp/test_pipe.joblib")
    s = Settings()
    assert s.feature_pipeline_path == "/tmp/test_pipe.joblib"


def test_get_settings_returns_fresh_instance():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 == s2


def test_is_debug_false_by_default(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert Settings().is_debug is False


def test_is_debug_true_when_debug(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert Settings().is_debug is True


def test_cors_origins_list_single(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://example.com")
    s = Settings()
    assert s.cors_origins_list == ["https://example.com"]


def test_cors_origins_list_multiple(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
    s = Settings()
    assert len(s.cors_origins_list) == 2


def test_enable_json_logs_default(monkeypatch):
    monkeypatch.delenv("ENABLE_JSON_LOGS", raising=False)
    assert Settings().enable_json_logs is False


def test_enable_json_logs_true(monkeypatch):
    monkeypatch.setenv("ENABLE_JSON_LOGS", "true")
    assert Settings().enable_json_logs is True
