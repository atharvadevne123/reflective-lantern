"""Tests for ticket-oracle app/config.py."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.config import Settings, get_settings


class TestSettings:
    def test_default_rate_limit(self) -> None:
        s = Settings()
        assert s.api_rate_limit == "60/minute"

    def test_default_max_batch_size(self) -> None:
        s = Settings()
        assert s.max_batch_size == 100

    def test_default_cache_maxsize(self) -> None:
        s = Settings()
        assert s.cache_maxsize == 512

    def test_default_cache_ttl(self) -> None:
        s = Settings()
        assert s.cache_ttl_seconds == 300.0

    def test_request_timeout_positive(self) -> None:
        s = Settings()
        assert s.request_timeout_seconds > 0

    def test_env_override_max_batch_size(self) -> None:
        with patch.dict(os.environ, {"MAX_BATCH_SIZE": "50"}):
            s = Settings()
            assert s.max_batch_size == 50

    def test_env_override_cache_maxsize(self) -> None:
        with patch.dict(os.environ, {"CACHE_MAXSIZE": "256"}):
            s = Settings()
            assert s.cache_maxsize == 256


class TestGetSettings:
    def test_returns_settings(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
