"""Tests for centralized settings configuration."""

from __future__ import annotations

import os

import pytest


def test_get_settings_returns_singleton():
    from app.config import get_settings

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_defaults():
    from app.config import get_settings

    s = get_settings()
    assert s.log_level == "INFO"
    assert s.rate_limit_rpm == 60
    assert s.drift_window_hours == 24
    assert s.drift_p_threshold == 0.05


def test_settings_model_version_default():
    from app.config import get_settings

    s = get_settings()
    assert s.model_version == "1.0.0"


def test_settings_cors_origins_is_list():
    from app.config import get_settings

    s = get_settings()
    assert isinstance(s.cors_origins, list)
    assert len(s.cors_origins) >= 1


def test_settings_faiss_defaults():
    from app.config import get_settings

    s = get_settings()
    assert s.faiss_anomaly_threshold > 0
    assert s.faiss_k_neighbours >= 1


def test_settings_lag_and_rolling_defaults():
    from app.config import get_settings

    s = get_settings()
    assert s.lag_periods == 2
    assert s.rolling_window == 5
