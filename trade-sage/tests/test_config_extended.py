"""Extended tests for trade-sage app/config.py."""

from __future__ import annotations

import pytest

from app.config import AppConfig, load_config


class TestAppConfigDefaults:
    def test_load_returns_instance(self) -> None:
        assert isinstance(load_config(), AppConfig)

    def test_default_log_level(self) -> None:
        cfg = load_config()
        assert cfg.log_level == "INFO"

    def test_default_drift_threshold(self) -> None:
        cfg = load_config()
        assert 0.0 < cfg.drift_threshold < 1.0

    def test_default_cors_origins_list(self) -> None:
        cfg = load_config()
        assert isinstance(cfg.cors_origins, list)
        assert len(cfg.cors_origins) >= 1

    def test_default_horizon_days(self) -> None:
        cfg = load_config()
        assert cfg.horizon_days > 0

    def test_frozen(self) -> None:
        cfg = load_config()
        with pytest.raises(Exception):
            cfg.log_level = "DEBUG"  # type: ignore[misc]


class TestAppConfigEnvOverride:
    def test_log_level_override(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        cfg = load_config()
        assert cfg.log_level == "DEBUG"

    def test_horizon_days_override(self, monkeypatch) -> None:
        monkeypatch.setenv("HORIZON_DAYS", "10")
        cfg = load_config()
        assert cfg.horizon_days == 10

    def test_cors_multiple_origins(self, monkeypatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com,http://b.com")
        cfg = load_config()
        assert len(cfg.cors_origins) == 2

    def test_cache_ttl_override(self, monkeypatch) -> None:
        monkeypatch.setenv("CACHE_TTL_SECONDS", "600")
        cfg = load_config()
        assert cfg.cache_ttl_seconds == pytest.approx(600.0)
