"""Extended tests for ops-vision app/config.py."""

from __future__ import annotations

import pytest

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
        with pytest.raises(Exception):
            Settings(drift_threshold=0.0)

    def test_invalid_drift_threshold_one(self) -> None:
        with pytest.raises(Exception):
            Settings(drift_threshold=1.0)

    def test_invalid_rate_limit_zero(self) -> None:
        with pytest.raises(Exception):
            Settings(rate_limit_requests=0)

    def test_invalid_rate_limit_negative(self) -> None:
        with pytest.raises(Exception):
            Settings(rate_limit_requests=-1)


class TestGetSettings:
    def test_returns_settings_instance(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
