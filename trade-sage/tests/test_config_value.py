"""Tests for get_config_value and set_config_override."""

from __future__ import annotations

import pytest

from app.config import get_config_value, reset_config, set_config_override


class TestGetConfigValue:
    def setup_method(self) -> None:
        reset_config()

    def test_returns_known_field(self) -> None:
        result = get_config_value("log_level")
        assert isinstance(result, str)

    def test_returns_default_for_missing_key(self) -> None:
        result = get_config_value("nonexistent_field", default="fallback")
        assert result == "fallback"

    def test_returns_none_default_when_not_given(self) -> None:
        result = get_config_value("no_such_key")
        assert result is None

    def test_rate_limit_is_int(self) -> None:
        result = get_config_value("api_rate_limit")
        assert isinstance(result, int)
        assert result > 0

    def test_cache_ttl_is_float(self) -> None:
        result = get_config_value("cache_ttl_seconds")
        assert isinstance(result, float)


class TestSetConfigOverride:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_overrides_log_level(self) -> None:
        set_config_override("log_level", "DEBUG")
        assert get_config_value("log_level") == "DEBUG"

    def test_overrides_rate_limit(self) -> None:
        set_config_override("api_rate_limit", 999)
        assert get_config_value("api_rate_limit") == 999

    def test_raises_for_invalid_key(self) -> None:
        with pytest.raises(AttributeError, match="no field"):
            set_config_override("made_up_key", "value")

    def test_other_fields_unchanged_after_override(self) -> None:
        original_ttl = get_config_value("cache_ttl_seconds")
        set_config_override("log_level", "WARNING")
        assert get_config_value("cache_ttl_seconds") == original_ttl
