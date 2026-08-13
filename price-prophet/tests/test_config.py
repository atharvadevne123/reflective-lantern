"""Tests for app/config.py."""

from __future__ import annotations


def test_settings_defaults():
    from app.config import Settings

    s = Settings()
    assert s.app_name == "Price-Prophet"
    assert s.debug is False
    assert s.cache_ttl_seconds == 300


def test_settings_min_price_multiplier():
    from app.config import Settings

    s = Settings()
    assert 0 < s.min_price_multiplier < 1.0


def test_settings_max_price_multiplier():
    from app.config import Settings

    s = Settings()
    assert s.max_price_multiplier > 1.0


def test_settings_singleton_importable():
    from app.config import settings

    assert settings is not None
    assert settings.app_name == "Price-Prophet"


def test_settings_log_level():
    from app.config import settings

    assert settings.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")


def test_settings_default_elasticity_negative():
    from app.config import settings

    assert settings.default_elasticity < 0
