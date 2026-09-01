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


def test_settings_version_is_string() -> None:
    from app.config import settings

    assert isinstance(settings.version, str)
    assert len(settings.version) > 0


def test_settings_max_multiplier_greater_than_min() -> None:
    from app.config import settings

    assert settings.max_price_multiplier > settings.min_price_multiplier


def test_settings_cache_ttl_positive() -> None:
    from app.config import settings

    assert settings.cache_ttl_seconds > 0


def test_settings_app_name_is_string() -> None:
    from app.config import settings

    assert isinstance(settings.app_name, str)
    assert len(settings.app_name) > 0


def test_settings_version_format() -> None:
    from app.config import settings

    parts = settings.version.split(".")
    assert len(parts) >= 1
    assert all(p.isdigit() for p in parts)


def test_settings_multipliers_sane_range() -> None:
    from app.config import settings

    assert settings.min_price_multiplier > 0.0
    assert settings.max_price_multiplier < 10.0


def test_settings_model_dir_is_string() -> None:
    from app.config import settings

    assert isinstance(settings.model_dir, str)


def test_settings_debug_is_bool() -> None:
    from app.config import settings

    assert isinstance(settings.debug, bool)


import pytest


@pytest.mark.parametrize(
    "attr",
    ["min_price_multiplier", "max_price_multiplier", "model_dir", "debug"],
)
def test_settings_has_required_attribute(attr: str) -> None:
    from app.config import settings

    assert hasattr(settings, attr)


def test_settings_min_less_than_max() -> None:
    from app.config import settings

    assert settings.min_price_multiplier < settings.max_price_multiplier
