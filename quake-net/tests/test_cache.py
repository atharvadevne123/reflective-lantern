"""Tests for the TTL/LRU response cache."""

from __future__ import annotations

import pytest

from app.cache import TTLCache, get_cache


class TestTTLCacheBasics:
    def test_get_missing_key_returns_none(self) -> None:
        assert TTLCache().get("absent") is None

    def test_set_then_get_roundtrip(self) -> None:
        cache = TTLCache()
        cache.set("k", {"value": 1})
        assert cache.get("k") == {"value": 1}

    def test_overwrite_replaces_value(self) -> None:
        cache = TTLCache()
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"

    def test_len_reflects_entry_count(self) -> None:
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2

    def test_clear_empties_cache(self) -> None:
        cache = TTLCache()
        cache.set("a", 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_falsy_values_are_retrievable(self) -> None:
        cache = TTLCache()
        cache.set("zero", 0)
        assert cache.get("zero") == 0


class TestTTLExpiry:
    def test_expired_entry_returns_none(self) -> None:
        cache = TTLCache(ttl_seconds=0.0)
        cache.set("k", "v")
        assert cache.get("k") is None

    def test_expired_entry_is_evicted(self) -> None:
        cache = TTLCache(ttl_seconds=0.0)
        cache.set("k", "v")
        cache.get("k")
        assert len(cache) == 0

    def test_live_entry_survives(self) -> None:
        cache = TTLCache(ttl_seconds=60.0)
        cache.set("k", "v")
        assert cache.get("k") == "v"


class TestLRUEviction:
    def test_max_entries_enforced(self) -> None:
        cache = TTLCache(max_entries=3)
        for i in range(6):
            cache.set(f"k{i}", i)
        assert len(cache) == 3

    def test_oldest_entry_evicted_first(self) -> None:
        cache = TTLCache(max_entries=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("c") == 3

    def test_recent_read_protects_entry(self) -> None:
        cache = TTLCache(max_entries=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # promote "a" to most-recently-used
        cache.set("c", 3)
        assert cache.get("a") == 1
        assert cache.get("b") is None


class TestCacheStats:
    def test_hits_and_misses_counted(self) -> None:
        cache = TTLCache()
        cache.set("k", "v")
        cache.get("k")
        cache.get("absent")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_hit_rate_computed(self) -> None:
        cache = TTLCache()
        cache.set("k", "v")
        cache.get("k")
        cache.get("k")
        cache.get("absent")
        assert cache.stats()["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_hit_rate_zero_when_untouched(self) -> None:
        assert TTLCache().stats()["hit_rate"] == 0.0

    def test_stats_report_configuration(self) -> None:
        stats = TTLCache(ttl_seconds=42.0, max_entries=9).stats()
        assert stats["ttl_seconds"] == 42.0
        assert stats["max_entries"] == 9

    def test_clear_resets_counters(self) -> None:
        cache = TTLCache()
        cache.get("absent")
        cache.clear()
        assert cache.stats()["misses"] == 0


class TestModuleCache:
    def test_get_cache_is_singleton(self) -> None:
        assert get_cache() is get_cache()
