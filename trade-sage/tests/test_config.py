"""Tests for centralised application configuration."""

from __future__ import annotations

import pytest

from app.config import AppConfig, load_config


def test_load_config_returns_app_config():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)


def test_default_database_url():
    cfg = load_config()
    assert "sqlite" in cfg.database_url


def test_default_log_level():
    cfg = load_config()
    assert cfg.log_level == "INFO"


def test_default_cache_ttl():
    cfg = load_config()
    assert cfg.cache_ttl_seconds == 300.0


def test_default_cache_maxsize():
    cfg = load_config()
    assert cfg.cache_maxsize == 512


def test_default_drift_threshold():
    cfg = load_config()
    assert cfg.drift_threshold == 0.05


def test_default_horizon_days():
    cfg = load_config()
    assert cfg.horizon_days == 5


def test_cors_origins_is_list():
    cfg = load_config()
    assert isinstance(cfg.cors_origins, list)
    assert len(cfg.cors_origins) >= 1


def test_env_override_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/trade_sage")
    cfg = load_config()
    assert "postgresql" in cfg.database_url


def test_env_override_rate_limit(monkeypatch):
    monkeypatch.setenv("API_RATE_LIMIT", "100")
    cfg = load_config()
    assert cfg.api_rate_limit == 100


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises((AttributeError, TypeError)):
        cfg.log_level = "DEBUG"  # type: ignore[misc]
