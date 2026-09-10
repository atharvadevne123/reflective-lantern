"""Tests for configuration loading and validation."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_get_settings_returns_settings():
    from app.config import get_settings, Settings
    s = get_settings()
    assert isinstance(s, Settings)


def test_settings_has_database_url():
    from app.config import get_settings
    s = get_settings()
    assert isinstance(s.database_url, str)
    assert len(s.database_url) > 0


def test_settings_drift_threshold_positive():
    from app.config import get_settings
    s = get_settings()
    assert s.drift_ks_threshold > 0


def test_settings_window_days_positive():
    from app.config import get_settings
    s = get_settings()
    assert s.reference_window_days > 0
    assert s.monitor_window_days > 0


def test_settings_singleton():
    from app.config import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
