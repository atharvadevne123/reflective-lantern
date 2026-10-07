"""TTL cache tests for Sports-Oracle."""

from __future__ import annotations

import time

import pytest


def test_cache_miss_returns_none():
    from app.cache import TTLCache

    cache = TTLCache(maxsize=10, ttl=60.0)
    assert cache.get({"key": "value"}) is None


def test_cache_set_and_get():
    from app.cache import TTLCache

    cache = TTLCache(maxsize=10, ttl=60.0)
    features = {"home_form": 0.7, "away_form": 0.4}
    cache.set(features, {"predicted_outcome": "H"})
    result = cache.get(features)
    assert result["predicted_outcome"] == "H"


def test_cache_expiry():
    from app.cache import TTLCache

    cache = TTLCache(maxsize=10, ttl=0.05)
    features = {"home_form": 0.7}
    cache.set(features, "value")
    time.sleep(0.1)
    assert cache.get(features) is None


def test_cache_evicts_when_full():
    from app.cache import TTLCache

    cache = TTLCache(maxsize=3, ttl=60.0)
    for i in range(4):
        cache.set({"idx": i}, f"val_{i}")
    assert len(cache) <= 3


def test_cache_clear():
    from app.cache import TTLCache

    cache = TTLCache(maxsize=10, ttl=60.0)
    cache.set({"a": 1}, "x")
    cache.clear()
    assert len(cache) == 0


def test_cache_key_deterministic():
    from app.cache import TTLCache

    cache = TTLCache(maxsize=10, ttl=60.0)
    features = {"b": 2, "a": 1}
    cache.set(features, "result")
    # Same dict in different order should hit the cache
    assert cache.get({"a": 1, "b": 2}) == "result"
