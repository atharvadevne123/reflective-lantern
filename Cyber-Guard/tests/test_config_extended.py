"""Extended tests for Cyber-Guard app/config.py."""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings


class TestSettingsDefaults:
    def test_creates_instance(self) -> None:
        assert Settings() is not None

    def test_default_log_level(self) -> None:
        s = Settings()
        assert s.log_level == "INFO"

    def test_default_drift_threshold(self) -> None:
        s = Settings()
        assert 0.0 < s.drift_p_threshold < 1.0

    def test_default_rate_limit(self) -> None:
        s = Settings()
        assert s.rate_limit_per_minute > 0

    def test_is_sqlite_default(self) -> None:
        s = Settings()
        assert s.is_sqlite() is True

    def test_frozen_immutable(self) -> None:
        s = Settings()
        with pytest.raises(Exception):
            s.log_level = "DEBUG"  # type: ignore[misc]


class TestSettingsEnvOverride:
    def test_log_level_env(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        s = Settings()
        assert s.log_level == "WARNING"

    def test_drift_threshold_env(self, monkeypatch) -> None:
        monkeypatch.setenv("DRIFT_P_THRESHOLD", "0.01")
        s = Settings()
        assert s.drift_p_threshold == pytest.approx(0.01)

    def test_invalid_float_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("DRIFT_P_THRESHOLD", "not_a_float")
        s = Settings()
        assert s.drift_p_threshold == pytest.approx(0.05)

    def test_invalid_int_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "oops")
        s = Settings()
        assert s.rate_limit_per_minute == 120


class TestGetSettings:
    def test_returns_settings_instance(self) -> None:
        assert isinstance(get_settings(), Settings)

    def test_returns_fresh_instance_each_call(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is not s2
