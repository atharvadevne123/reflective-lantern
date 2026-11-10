"""Extended tests for cart-mind app/cache.py helpers."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache, cache_hit_rate, evict_expired, warm_cache


@pytest.fixture()
def cache() -> TTLCache:
    return TTLCache(ttl_seconds=10, max_size=100)


class TestCacheHitRate:
    def test_zero_on_no_requests(self, cache: TTLCache) -> None:
        assert cache_hit_rate(cache) == pytest.approx(0.0)

    def test_one_on_all_hits(self, cache: TTLCache) -> None:
        cache.set("k", "v")
        cache.get("k")
        assert cache_hit_rate(cache) == pytest.approx(1.0)

    def test_zero_on_all_misses(self, cache: TTLCache) -> None:
        cache.get("missing")
        assert cache_hit_rate(cache) == pytest.approx(0.0)

    def test_partial_hit_rate(self, cache: TTLCache) -> None:
        cache.set("k", "v")
        cache.get("k")
        cache.get("missing")
        rate = cache_hit_rate(cache)
        assert 0.0 < rate < 1.0


class TestWarmCache:
    def test_inserts_all_items(self, cache: TTLCache) -> None:
        items = {"a": 1, "b": 2, "c": 3}
        count = warm_cache(cache, items)
        assert count == 3

    def test_items_retrievable_after_warm(self, cache: TTLCache) -> None:
        warm_cache(cache, {"key": "value"})
        assert cache.get("key") == "value"

    def test_empty_dict_inserts_zero(self, cache: TTLCache) -> None:
        count = warm_cache(cache, {})
        assert count == 0

    def test_returns_count(self, cache: TTLCache) -> None:
        count = warm_cache(cache, {"x": 1, "y": 2})
        assert count == 2


class TestEvictExpired:
    def test_evicts_expired_entries(self) -> None:
        short_cache = TTLCache(ttl_seconds=1, max_size=100)
        short_cache.set("k", "v")
        time.sleep(1.1)
        evicted = evict_expired(short_cache)
        assert evicted >= 1

    def test_keeps_fresh_entries(self, cache: TTLCache) -> None:
        cache.set("k", "v")
        evicted = evict_expired(cache)
        assert evicted == 0
        assert cache.get("k") == "v"

    def test_empty_cache_no_eviction(self, cache: TTLCache) -> None:
        assert evict_expired(cache) == 0
