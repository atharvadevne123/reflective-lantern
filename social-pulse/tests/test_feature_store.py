"""Tests for the in-memory feature cache."""

from __future__ import annotations

import time

import pytest

from app.feature_store import FeatureCache


class TestFeatureCacheGet:
    def test_miss_on_empty_cache(self) -> None:
        cache = FeatureCache()
        assert cache.get("nonexistent") is None

    def test_hit_after_set(self) -> None:
        cache = FeatureCache()
        cache.set("k1", {"a": 1})
        assert cache.get("k1") == {"a": 1}

    def test_expired_entry_returns_none(self) -> None:
        cache = FeatureCache(ttl_seconds=0.01)
        cache.set("k2", {"b": 2})
        time.sleep(0.02)
        assert cache.get("k2") is None


class TestFeatureCacheSet:
    def test_size_increments(self) -> None:
        cache = FeatureCache()
        cache.set("a", {})
        cache.set("b", {})
        assert cache.size == 2

    def test_overwrite_existing(self) -> None:
        cache = FeatureCache()
        cache.set("x", {"v": 1})
        cache.set("x", {"v": 2})
        assert cache.get("x") == {"v": 2}
        assert cache.size == 1

    def test_evicts_oldest_when_full(self) -> None:
        cache = FeatureCache(max_size=2)
        cache.set("first", {})
        cache.set("second", {})
        cache.set("third", {})
        assert cache.size == 2
        assert cache.get("first") is None


class TestFeatureCacheInvalidate:
    def test_removes_existing_key(self) -> None:
        cache = FeatureCache()
        cache.set("k", {"v": 42})
        result = cache.invalidate("k")
        assert result is True
        assert cache.get("k") is None

    def test_returns_false_for_missing_key(self) -> None:
        cache = FeatureCache()
        assert cache.invalidate("absent") is False


class TestFeatureCacheClear:
    def test_clears_all_entries(self) -> None:
        cache = FeatureCache()
        for i in range(5):
            cache.set(str(i), {})
        cache.clear()
        assert cache.size == 0

    def test_resets_stats(self) -> None:
        cache = FeatureCache()
        cache.set("k", {})
        cache.get("k")
        cache.clear()
        assert cache.hit_rate == 0.0


class TestFeatureCacheStats:
    def test_stats_has_required_keys(self) -> None:
        cache = FeatureCache()
        s = cache.stats()
        assert "size" in s
        assert "hits" in s
        assert "misses" in s
        assert "hit_rate" in s

    def test_hit_rate_zero_on_empty(self) -> None:
        cache = FeatureCache()
        assert cache.hit_rate == 0.0

    def test_hit_rate_after_hits(self) -> None:
        cache = FeatureCache()
        cache.set("k", {"v": 1})
        cache.get("k")
        cache.get("k")
        assert cache.hit_rate == 1.0

    def test_hit_rate_mixed(self) -> None:
        cache = FeatureCache()
        cache.set("k", {"v": 1})
        cache.get("k")  # hit
        cache.get("miss")  # miss
        assert abs(cache.hit_rate - 0.5) < 0.01


@pytest.mark.parametrize("max_size", [1, 5, 10, 100])
def test_cache_respects_max_size(max_size: int) -> None:
    cache = FeatureCache(max_size=max_size)
    for i in range(max_size + 5):
        cache.set(str(i), {"i": i})
    assert cache.size <= max_size


class TestGetAllKeys:
    def test_empty_cache_returns_empty_list(self) -> None:
        cache = FeatureCache()
        assert cache.get_all_keys() == []

    def test_returns_all_keys(self) -> None:
        cache = FeatureCache()
        cache.set("alpha", {})
        cache.set("beta", {})
        keys = cache.get_all_keys()
        assert "alpha" in keys
        assert "beta" in keys

    def test_length_matches_size(self) -> None:
        cache = FeatureCache()
        for i in range(5):
            cache.set(str(i), {})
        assert len(cache.get_all_keys()) == cache.size

    def test_clear_empties_keys(self) -> None:
        cache = FeatureCache()
        cache.set("k", {})
        cache.clear()
        assert cache.get_all_keys() == []

    def test_returns_list_type(self) -> None:
        cache = FeatureCache()
        assert isinstance(cache.get_all_keys(), list)


class TestFeatureCacheWarm:
    def test_warm_inserts_entries(self) -> None:
        cache = FeatureCache(max_size=100)
        entries = {f"key{i}": {"i": i} for i in range(10)}
        inserted = cache.warm(entries)
        assert inserted == 10

    def test_warm_returns_count(self) -> None:
        cache = FeatureCache(max_size=5)
        entries = {f"k{i}": {} for i in range(20)}
        inserted = cache.warm(entries)
        assert inserted <= 5

    def test_warm_entries_retrievable(self) -> None:
        cache = FeatureCache()
        cache.warm({"abc": {"feature": 42}})
        assert cache.get("abc") == {"feature": 42}

    def test_warm_empty_dict(self) -> None:
        cache = FeatureCache()
        inserted = cache.warm({})
        assert inserted == 0

    def test_warm_respects_max_size(self) -> None:
        cache = FeatureCache(max_size=3)
        entries = {f"k{i}": {} for i in range(10)}
        cache.warm(entries)
        assert cache.size <= 3
