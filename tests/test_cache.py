"""TTL cache tests."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache


def test_set_and_get():
    c = TTLCache(ttl_seconds=10)
    c.set("k", 42)
    assert c.get("k") == 42

    def test_miss_returns_none(self):
        c = TTLCache()
        assert c.get("nonexistent") is None

def test_missing_key():
    c = TTLCache(ttl_seconds=10)
    assert c.get("nonexistent") is None

    def test_hit_rate_tracking(self):
        c = TTLCache(default_ttl=60.0)
        c.set("x", 42)
        c.get("x")
        c.get("x")
        c.get("missing")
        assert c.hits == 2
        assert c.misses == 1
        assert c.hit_rate == pytest.approx(2 / 3, rel=0.01)

def test_ttl_expiry():
    c = TTLCache(ttl_seconds=0)
    c.set("k", "v")
    time.sleep(0.01)
    assert c.get("k") is None

    def test_clear_empties_cache(self):
        c = TTLCache()
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.size == 0
        assert c.hits == 0
        assert c.misses == 0

def test_invalidate():
    c = TTLCache(ttl_seconds=60)
    c.set("k", 1)
    c.invalidate("k")
    assert c.get("k") is None

    def test_evict_expired(self):
        c = TTLCache(default_ttl=0.05)
        c.set("x", 1)
        c.set("y", 2)
        time.sleep(0.1)
        c.set("z", 3, ttl=60.0)
        evicted = c.evict_expired()
        assert evicted == 2
        assert c.size == 1

def test_clear():
    c = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert len(c) == 0


def test_max_size_eviction():
    c = TTLCache(ttl_seconds=60, max_size=3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.set("d", 4)
    assert len(c) == 3


@pytest.mark.parametrize("value", [None, 0, "", [], {"x": 1}])
def test_stores_falsy_values(value):
    c = TTLCache(ttl_seconds=60)
    c.set("k", value)
    assert c.get("k") == value
