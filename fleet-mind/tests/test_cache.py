"""Tests for the prediction cache module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.cache import (
    _make_cache_key,
    cache_size,
    clear_cache,
    get_cached_prediction,
    set_cached_prediction,
)


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


class TestCacheKey:
    def test_same_reading_same_key(self):
        r = {"speed_kmh": 80.0, "engine_temp_c": 90.0}
        assert _make_cache_key(r) == _make_cache_key(r)

    def test_different_reading_different_key(self):
        r1 = {"speed_kmh": 80.0}
        r2 = {"speed_kmh": 81.0}
        assert _make_cache_key(r1) != _make_cache_key(r2)

    def test_key_is_16_chars(self):
        assert len(_make_cache_key({"x": 1.0})) == 16


class TestCacheOperations:
    def test_miss_returns_none(self):
        assert get_cached_prediction({"speed_kmh": 99.9}) is None

    def test_set_then_get(self):
        reading = {"speed_kmh": 80.0, "engine_temp_c": 90.0}
        result = {"maintenance_needed": False, "anomaly_score": 0.1}
        set_cached_prediction(reading, result)
        assert get_cached_prediction(reading) == result

    def test_cache_size_increments(self):
        assert cache_size() == 0
        set_cached_prediction({"a": 1.0}, {"x": 1})
        assert cache_size() == 1

    def test_clear_cache_resets_size(self):
        set_cached_prediction({"a": 1.0}, {"x": 1})
        clear_cache()
        assert cache_size() == 0

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_multiple_entries(self, n):
        for i in range(n):
            set_cached_prediction({"v": float(i)}, {"result": i})
        assert cache_size() == n
