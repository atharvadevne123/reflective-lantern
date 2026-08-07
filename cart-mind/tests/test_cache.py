"""Tests for the TTL cache."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache


class TestTTLCacheBasics:
    def test_set_then_get_returns_value(self):
        cache = TTLCache()
        cache.set("k", {"a": 1})
        assert cache.get("k") == {"a": 1}

    def test_get_missing_key_returns_none(self):
        cache = TTLCache()
        assert cache.get("absent") is None

    def test_set_overwrites_existing_key(self):
        cache = TTLCache()
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"

    def test_clear_empties_cache(self):
        cache = TTLCache()
        cache.set("k", 1)
        cache.clear()
        assert cache.get("k") is None
        assert cache.stats()["entries"] == 0


class TestTTLExpiry:
    def test_entry_expires_after_ttl(self):
        cache = TTLCache(ttl_seconds=0.05)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.08)
        assert cache.get("k") is None

    def test_entry_survives_within_ttl(self):
        cache = TTLCache(ttl_seconds=5.0)
        cache.set("k", "v")
        time.sleep(0.02)
        assert cache.get("k") == "v"

    def test_expired_entry_counts_as_miss(self):
        cache = TTLCache(ttl_seconds=0.05)
        cache.set("k", "v")
        time.sleep(0.08)
        cache.get("k")
        assert cache.stats()["misses"] >= 1


class TestLRUEviction:
    def test_evicts_when_over_capacity(self):
        cache = TTLCache(max_entries=3)
        for i in range(5):
            cache.set(f"k{i}", i)
        assert cache.stats()["entries"] == 3

    def test_evicts_least_recently_used_first(self):
        cache = TTLCache(max_entries=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")  # 'a' becomes most-recently-used, so 'b' is now the LRU
        cache.set("d", 4)
        assert cache.get("b") is None
        assert cache.get("a") == 1

    def test_recently_set_key_survives_eviction(self):
        cache = TTLCache(max_entries=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("c") == 3


class TestCacheStats:
    def test_hit_rate_all_hits(self):
        cache = TTLCache()
        cache.set("k", 1)
        cache.get("k")
        cache.get("k")
        assert cache.stats()["hit_rate"] == 1.0

    def test_hit_rate_all_misses(self):
        cache = TTLCache()
        cache.get("x")
        cache.get("y")
        assert cache.stats()["hit_rate"] == 0.0

    def test_hit_rate_mixed(self):
        cache = TTLCache()
        cache.set("k", 1)
        cache.get("k")
        cache.get("absent")
        assert cache.stats()["hit_rate"] == 0.5

    def test_stats_reports_config(self):
        cache = TTLCache(ttl_seconds=42.0, max_entries=7)
        stats = cache.stats()
        assert stats["ttl_seconds"] == 42.0
        assert stats["max_entries"] == 7

    def test_empty_cache_hit_rate_is_zero(self):
        assert TTLCache().stats()["hit_rate"] == 0.0

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_entry_count_tracks_inserts(self, n):
        cache = TTLCache(max_entries=1000)
        for i in range(n):
            cache.set(f"k{i}", i)
        assert cache.stats()["entries"] == n


class TestTTLCacheDeleteAndSize:
    """Tests for the delete() and size() methods added in the 2026-08-07 improvement run."""

    def test_delete_existing_key(self):
        cache = TTLCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None

    def test_delete_absent_key_returns_false(self):
        cache = TTLCache()
        assert cache.delete("nonexistent") is False

    def test_size_empty_cache(self):
        assert TTLCache().size() == 0

    def test_size_after_inserts(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size() == 2

    def test_size_decrements_after_delete(self):
        cache = TTLCache()
        cache.set("x", 10)
        cache.delete("x")
        assert cache.size() == 0

    def test_delete_then_reinsert(self):
        cache = TTLCache()
        cache.set("k", "v1")
        cache.delete("k")
        cache.set("k", "v2")
        assert cache.get("k") == "v2"

    @pytest.mark.parametrize("n_delete", [0, 1, 5])
    def test_size_after_partial_delete(self, n_delete):
        cache = TTLCache(max_entries=100)
        for i in range(10):
            cache.set(f"k{i}", i)
        for i in range(n_delete):
            cache.delete(f"k{i}")
        assert cache.size() == 10 - n_delete
