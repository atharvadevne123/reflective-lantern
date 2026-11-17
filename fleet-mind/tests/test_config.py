"""Tests for Fleet-Mind configuration settings."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings, get_settings


def test_settings_has_database_url():
    s = Settings()
    assert s.database_url


def test_settings_default_database_url_is_sqlite():
    s = Settings()
    assert "sqlite" in s.database_url.lower()


def test_settings_default_drift_threshold():
    s = Settings()
    assert 0.0 < s.drift_threshold < 1.0


def test_settings_default_drift_window():
    s = Settings()
    assert s.drift_window > 0


def test_settings_default_api_title():
    s = Settings()
    assert s.api_title


def test_settings_default_rate_limit():
    s = Settings()
    assert s.rate_limit_requests > 0


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("DRIFT_THRESHOLD", "0.1")
    s = Settings()
    assert abs(s.drift_threshold - 0.1) < 0.001


def test_settings_log_level_default():
    s = Settings()
    assert s.log_level in ("INFO", "DEBUG", "WARNING", "ERROR")


def test_settings_log_json_default():
    s = Settings()
    assert isinstance(s.log_json, bool)


def test_get_settings_returns_cached_instance():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_get_settings_returns_settings_instance():
    s = get_settings()
    assert isinstance(s, Settings)


@pytest.mark.parametrize("field", ["database_url", "model_path", "metrics_path", "api_title"])
def test_required_string_fields_are_strings(field):
    s = Settings()
    assert isinstance(getattr(s, field), str)
