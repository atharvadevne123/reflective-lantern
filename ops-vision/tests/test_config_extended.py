"""Extended tests for ops-vision app/config.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


class TestSettingsDefaults:
    def test_creates_instance(self) -> None:
        assert Settings() is not None

    def test_default_app_name(self) -> None:
        s = Settings()
        assert s.app_name == "Ops-Vision"

    def test_default_log_level(self) -> None:
        s = Settings()
        assert s.log_level == "INFO"

    def test_default_debug_false(self) -> None:
        s = Settings()
        assert s.debug is False

    def test_default_embedding_dim_positive(self) -> None:
        s = Settings()
        assert s.embedding_dim > 0

    def test_default_drift_threshold_valid(self) -> None:
        s = Settings()
        assert 0.0 < s.drift_threshold < 1.0

    def test_default_rate_limit_positive(self) -> None:
        s = Settings()
        assert s.rate_limit_requests > 0


class TestSettingsValidation:
    def test_invalid_drift_threshold_zero(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            Settings(drift_threshold=0.0)

    def test_invalid_drift_threshold_one(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            Settings(drift_threshold=1.0)

    def test_invalid_rate_limit_zero(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            Settings(rate_limit_requests=0)

    def test_invalid_rate_limit_negative(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            Settings(rate_limit_requests=-1)


class TestGetSettings:
    def test_returns_settings_instance(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestSettingsFieldTypes:
    def test_embedding_dim_is_int(self) -> None:
        s = Settings()
        assert isinstance(s.embedding_dim, int)

    def test_reference_window_size_positive(self) -> None:
        s = Settings()
        assert s.reference_window_size > 0

    def test_current_window_size_positive(self) -> None:
        s = Settings()
        assert s.current_window_size > 0

    def test_rate_limit_window_seconds_is_int(self) -> None:
        s = Settings()
        assert isinstance(s.rate_limit_window_seconds, int)

    def test_debug_default_false(self) -> None:
        s = Settings()
        assert s.debug is False

    def test_log_level_default(self) -> None:
        s = Settings()
        assert s.log_level in ("INFO", "DEBUG", "WARNING", "ERROR")
