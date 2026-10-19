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
        cache.get("k")   # hit
        cache.get("miss")  # miss
        assert abs(cache.hit_rate - 0.5) < 0.01


@pytest.mark.parametrize("max_size", [1, 5, 10, 100])
def test_cache_respects_max_size(max_size: int) -> None:
    cache = FeatureCache(max_size=max_size)
    for i in range(max_size + 5):
        cache.set(str(i), {"i": i})
    assert cache.size <= max_size
