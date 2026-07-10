"""Tests for TTL cache."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache


class TestTTLCache:
    def test_set_and_get(self):
        c = TTLCache(default_ttl=10.0)
        c.set("k1", "value1")
        assert c.get("k1") == "value1"

    def test_miss_returns_none(self):
        c = TTLCache()
        assert c.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        c = TTLCache(default_ttl=0.05)
        c.set("k", "v")
        time.sleep(0.1)
        assert c.get("k") is None

    def test_hit_rate_tracking(self):
        c = TTLCache(default_ttl=60.0)
        c.set("x", 42)
        c.get("x")
        c.get("x")
        c.get("missing")
        assert c.hits == 2
        assert c.misses == 1
        assert c.hit_rate == pytest.approx(2 / 3, rel=0.01)

    def test_delete_removes_entry(self):
        c = TTLCache()
        c.set("key", "val")
        c.delete("key")
        assert c.get("key") is None

    def test_clear_empties_cache(self):
        c = TTLCache()
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.size == 0
        assert c.hits == 0
        assert c.misses == 0

    def test_size_reflects_entries(self):
        c = TTLCache()
        for i in range(5):
            c.set(str(i), i)
        assert c.size == 5

    def test_evict_expired(self):
        c = TTLCache(default_ttl=0.05)
        c.set("x", 1)
        c.set("y", 2)
        time.sleep(0.1)
        c.set("z", 3, ttl=60.0)
        evicted = c.evict_expired()
        assert evicted == 2
        assert c.size == 1

    @pytest.mark.parametrize("ttl", [0.05, 0.1, 0.2])
    def test_custom_ttl(self, ttl):
        c = TTLCache()
        c.set("k", "v", ttl=ttl)
        assert c.get("k") == "v"
        time.sleep(ttl + 0.05)
        assert c.get("k") is None
