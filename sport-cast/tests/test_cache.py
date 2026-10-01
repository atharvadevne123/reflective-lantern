"""TTL/LRU cache tests."""
from __future__ import annotations

import time

import pytest

from app.cache import TTLCache


def test_cache_miss_returns_none():
    cache = TTLCache(ttl=60)
    assert cache.get({"a": 1}) is None


def test_cache_hit_returns_value():
    cache = TTLCache(ttl=60)
    payload = {"match_id": "x", "home_elo": 1500}
    cache.set(payload, {"result": "home_win"})
    assert cache.get(payload) == {"result": "home_win"}


def test_cache_ttl_expiry():
    cache = TTLCache(ttl=1)
    payload = {"key": "expiry-test"}
    cache.set(payload, "value")
    time.sleep(1.05)
    assert cache.get(payload) is None


def test_cache_max_size_eviction():
    cache = TTLCache(ttl=60, max_size=3)
    for i in range(5):
        cache.set({"i": i}, f"value-{i}")
    assert cache.size <= 3


def test_cache_invalidate_removes_entry():
    cache = TTLCache(ttl=60)
    payload = {"test": "invalidate"}
    cache.set(payload, "result")
    removed = cache.invalidate(payload)
    assert removed is True
    assert cache.get(payload) is None


def test_cache_clear():
    cache = TTLCache(ttl=60)
    cache.set({"a": 1}, "x")
    cache.set({"b": 2}, "y")
    cache.clear()
    assert cache.size == 0


@pytest.mark.parametrize("n", [1, 10, 50])
def test_cache_size_matches_inserts(n):
    cache = TTLCache(ttl=60, max_size=100)
    for i in range(n):
        cache.set({"i": i}, i)
    assert cache.size == n
