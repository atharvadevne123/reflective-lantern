"""TTL cache tests."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache, cache_key_from_dict, cache_stats_summary


def test_set_and_get() -> None:
    c = TTLCache(ttl_seconds=10)
    c.set("k", 42)
    assert c.get("k") == 42


def test_missing_key() -> None:
    c = TTLCache(ttl_seconds=10)
    assert c.get("nonexistent") is None


def test_ttl_expiry() -> None:
    c = TTLCache(ttl_seconds=0)
    c.set("k", "v")
    time.sleep(0.01)
    assert c.get("k") is None


def test_invalidate() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", 1)
    c.invalidate("k")
    assert c.get("k") is None


def test_clear() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert c.size == 0
    assert c.hits == 0
    assert c.misses == 0


def test_max_size_eviction() -> None:
    c = TTLCache(ttl_seconds=60, max_size=3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.set("d", 4)
    assert c.size == 3


def test_hit_rate() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("x", 42)
    c.get("x")
    c.get("x")
    c.get("missing")
    assert c.hits == 2
    assert c.misses == 1
    assert c.hit_rate == pytest.approx(2 / 3, rel=0.01)


def test_evict_expired() -> None:
    c = TTLCache(ttl_seconds=0)
    c.set("x", 1)
    c.set("y", 2)
    time.sleep(0.01)
    removed = c.evict_expired()
    assert removed == 2
    assert c.size == 0


@pytest.mark.parametrize("value", [None, 0, "", [], {"x": 1}])
def test_stores_falsy_values(value: object) -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", value)
    assert c.get("k") == value


@pytest.mark.parametrize("ttl,size", [(5, 10), (60, 100), (300, 1000)])
def test_construction_params(ttl: int, size: int) -> None:
    c = TTLCache(ttl_seconds=ttl, max_size=size)
    assert c.size == 0


def test_get_or_set_returns_existing_value() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", "existing")
    assert c.get_or_set("k", "default") == "existing"


def test_get_or_set_stores_default_when_absent() -> None:
    c = TTLCache(ttl_seconds=60)
    result = c.get_or_set("new_key", "default_val")
    assert result == "default_val"
    assert c.get("new_key") == "default_val"


def test_stats_returns_expected_keys() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", "v")
    c.get("k")
    s = c.stats()
    assert {"size", "hits", "misses", "hit_rate"} <= set(s.keys())


def test_stats_empty_cache() -> None:
    c = TTLCache()
    stats = c.stats()
    assert stats["size"] == 0
    assert stats["hit_rate"] == 0.0


def test_eviction_count_initial_zero() -> None:
    c = TTLCache(ttl_seconds=60, max_size=10)
    assert c.eviction_count == 0


def test_eviction_count_increments_on_overflow() -> None:
    c = TTLCache(ttl_seconds=60, max_size=3)
    for i in range(5):
        c.set(f"key_{i}", i)
    assert c.eviction_count == 2


def test_eviction_count_resets_on_clear() -> None:
    c = TTLCache(ttl_seconds=60, max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # triggers eviction
    assert c.eviction_count == 1
    c.clear()
    assert c.eviction_count == 0


def test_stats_includes_evictions() -> None:
    c = TTLCache(ttl_seconds=60, max_size=2)
    c.set("x", 1)
    c.set("y", 2)
    c.set("z", 3)
    stats = c.stats()
    assert "evictions" in stats
    assert stats["evictions"] == 1


@pytest.mark.parametrize(
    "max_size,n_inserts,expected_evictions",
    [
        (5, 5, 0),
        (5, 6, 1),
        (3, 10, 7),
    ],
)
def test_eviction_count_parametrized(max_size, n_inserts, expected_evictions) -> None:
    c = TTLCache(ttl_seconds=60, max_size=max_size)
    for i in range(n_inserts):
        c.set(f"k_{i}", i)
    assert c.eviction_count == expected_evictions


def test_warm_cache_inserts_entries() -> None:
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    count = warm_cache(c, {"a": 1, "b": 2, "c": 3})
    assert count == 3
    assert c.get("a") == 1
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_warm_cache_empty_dict() -> None:
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    count = warm_cache(c, {})
    assert count == 0


def test_warm_cache_returns_insert_count() -> None:
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    entries = {f"key_{i}": i for i in range(10)}
    count = warm_cache(c, entries)
    assert count == 10


def test_warm_cache_overwrite_existing() -> None:
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    c.set("x", "old")
    warm_cache(c, {"x": "new"})
    assert c.get("x") == "new"


@pytest.mark.parametrize("n", [1, 5, 20])
def test_warm_cache_parametrized_count(n) -> None:
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    entries = {str(i): i * 10 for i in range(n)}
    assert warm_cache(c, entries) == n


def test_ttl_cache_get_returns_none_missing() -> None:
    from app.cache import TTLCache

    c = TTLCache(ttl_seconds=30, max_size=10)
    assert c.get("nonexistent") is None


def test_ttl_cache_set_and_get() -> None:
    from app.cache import TTLCache

    c = TTLCache(ttl_seconds=60, max_size=10)
    c.set("k", 42)
    assert c.get("k") == 42


def test_ttl_cache_size_increments() -> None:
    from app.cache import TTLCache

    c = TTLCache(ttl_seconds=60, max_size=100)
    assert c.size == 0
    c.set("a", 1)
    c.set("b", 2)
    assert c.size == 2


def test_ttl_cache_clear_resets_size() -> None:
    from app.cache import TTLCache

    c = TTLCache(ttl_seconds=60, max_size=10)
    c.set("x", 99)
    c.clear()
    assert c.size == 0


@pytest.mark.parametrize("ttl", [1, 30, 300, 3600])
def test_ttl_cache_accepts_various_ttls(ttl: int) -> None:
    from app.cache import TTLCache

    c = TTLCache(ttl_seconds=ttl, max_size=5)
    c.set("key", "val")
    assert c.get("key") == "val"


class TestTTLCacheGetOrSet:
    def test_sets_on_miss(self) -> None:
        from app.cache import TTLCache

        c = TTLCache(ttl_seconds=60)
        result = c.get_or_set("k", 42)
        assert result == 42
        assert c.get("k") == 42

    def test_returns_existing_on_hit(self) -> None:
        from app.cache import TTLCache

        c = TTLCache(ttl_seconds=60)
        c.set("k", 99)
        result = c.get_or_set("k", 0)
        assert result == 99

    def test_different_keys_isolated(self) -> None:
        from app.cache import TTLCache

        c = TTLCache(ttl_seconds=60)
        c.get_or_set("a", 1)
        c.get_or_set("b", 2)
        assert c.get("a") == 1
        assert c.get("b") == 2


class TestTTLCacheStats:
    def test_stats_keys(self) -> None:
        from app.cache import TTLCache

        c = TTLCache(ttl_seconds=60)
        s = c.stats()
        assert "size" in s and "hits" in s and "misses" in s and "hit_rate" in s

    def test_hit_rate_after_ops(self) -> None:
        from app.cache import TTLCache

        c = TTLCache(ttl_seconds=60)
        c.set("x", 1)
        c.get("x")  # hit
        c.get("y")  # miss
        assert c.hit_rate == pytest.approx(0.5)

    def test_eviction_count_increases(self) -> None:
        from app.cache import TTLCache

        c = TTLCache(ttl_seconds=60, max_size=2)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)  # triggers eviction
        assert c.eviction_count >= 1


class TestTTLCacheContains:
    def test_present(self) -> None:
        from app.cache import TTLCache

        c = TTLCache(ttl_seconds=60)
        c.set("key", "val")
        assert "key" in c

    def test_absent(self) -> None:
        from app.cache import TTLCache

        c = TTLCache(ttl_seconds=60)
        assert "nope" not in c


class TestBuildCacheKey:
    def test_basic_key(self) -> None:
        from app.cache import build_cache_key

        assert build_cache_key("user", 42, "data") == "user:42:data"

    def test_single_part(self) -> None:
        from app.cache import build_cache_key

        assert build_cache_key("only") == "only"

    def test_numeric_parts(self) -> None:
        from app.cache import build_cache_key

        result = build_cache_key(1, 2, 3)
        assert result == "1:2:3"


class TestCacheHitRate:
    def test_all_hits(self) -> None:
        from app.cache import cache_hit_rate

        assert cache_hit_rate(10, 0) == pytest.approx(1.0)

    def test_all_misses(self) -> None:
        from app.cache import cache_hit_rate

        assert cache_hit_rate(0, 10) == pytest.approx(0.0)

    def test_zero_total(self) -> None:
        from app.cache import cache_hit_rate

        assert cache_hit_rate(0, 0) == 0.0

    def test_negative_raises(self) -> None:
        from app.cache import cache_hit_rate

        with pytest.raises(ValueError):
            cache_hit_rate(-1, 5)

    def test_mixed(self) -> None:
        from app.cache import cache_hit_rate

        assert cache_hit_rate(3, 7) == pytest.approx(0.3)


class TestWarmCache:
    def test_inserts_items(self) -> None:
        from app.cache import TTLCache, warm_cache

        cache = TTLCache(ttl_seconds=60, max_size=100)
        count = warm_cache(cache, {"a": 1, "b": 2, "c": 3})
        assert count == 3
        assert cache.get("a") == 1

    def test_empty_items(self) -> None:
        from app.cache import TTLCache, warm_cache

        cache = TTLCache(ttl_seconds=60, max_size=100)
        assert warm_cache(cache, {}) == 0


class TestEvictExpiredKeys:
    def test_evicts_expired(self) -> None:
        import time

        from app.cache import TTLCache, evict_expired_keys

        cache = TTLCache(ttl_seconds=0, max_size=100)
        cache.set("x", 1)
        time.sleep(0.01)
        evicted = evict_expired_keys(cache)
        assert evicted >= 0

    def test_no_eviction_when_fresh(self) -> None:
        from app.cache import TTLCache, evict_expired_keys

        cache = TTLCache(ttl_seconds=3600, max_size=100)
        cache.set("fresh", 99)
        assert evict_expired_keys(cache) == 0


def test_contains_present_key() -> None:
    c = TTLCache(ttl_seconds=10)
    c.set("x", 99)
    assert "x" in c


def test_contains_missing_key() -> None:
    c = TTLCache(ttl_seconds=10)
    assert "missing" not in c


def test_cache_key_from_dict_basic() -> None:
    key = cache_key_from_dict({"b": 2, "a": 1})
    assert key == "a=1:b=2"


def test_cache_key_from_dict_prefix() -> None:
    key = cache_key_from_dict({"x": 5}, prefix="predict")
    assert key.startswith("predict:")


def test_cache_key_from_dict_empty() -> None:
    key = cache_key_from_dict({})
    assert key == ""


def test_cache_key_from_dict_deterministic() -> None:
    key1 = cache_key_from_dict({"z": 1, "a": 2})
    key2 = cache_key_from_dict({"a": 2, "z": 1})
    assert key1 == key2


def test_cache_stats_summary_keys() -> None:
    c = TTLCache(ttl_seconds=10)
    summary = cache_stats_summary(c)
    assert "hits" in summary
    assert "misses" in summary
    assert "evictions" in summary
    assert "hit_rate" in summary
    assert "size" in summary


def test_cache_stats_summary_empty_cache() -> None:
    c = TTLCache(ttl_seconds=10)
    summary = cache_stats_summary(c)
    assert summary["hits"] == 0
    assert summary["misses"] == 0
    assert summary["size"] == 0


def test_cache_stats_summary_after_hits() -> None:
    c = TTLCache(ttl_seconds=10)
    c.set("k", "v")
    c.get("k")
    c.get("k")
    summary = cache_stats_summary(c)
    assert summary["hits"] == 2
    assert summary["size"] == 1


def test_cache_stats_summary_hit_rate() -> None:
    c = TTLCache(ttl_seconds=10)
    c.set("k", "v")
    c.get("k")  # hit
    c.get("x")  # miss
    summary = cache_stats_summary(c)
    assert summary["hit_rate"] == pytest.approx(0.5)


class TestCacheFillRate:
    def test_empty_cache_fill_zero(self) -> None:
        from app.cache import cache_fill_rate

        c = TTLCache(ttl_seconds=10, max_size=10)
        assert cache_fill_rate(c) == pytest.approx(0.0)

    def test_half_full(self) -> None:
        from app.cache import cache_fill_rate

        c = TTLCache(ttl_seconds=60, max_size=10)
        for i in range(5):
            c.set(str(i), i)
        assert cache_fill_rate(c) == pytest.approx(0.5)

    def test_full_cache(self) -> None:
        from app.cache import cache_fill_rate

        c = TTLCache(ttl_seconds=60, max_size=3)
        for i in range(3):
            c.set(str(i), i)
        assert cache_fill_rate(c) == pytest.approx(1.0)

    def test_zero_max_size_returns_zero(self) -> None:
        from app.cache import cache_fill_rate

        c = TTLCache(ttl_seconds=10, max_size=0)
        assert cache_fill_rate(c) == pytest.approx(0.0)


class TestPeek:
    def test_existing_key_returns_value(self) -> None:
        from app.cache import peek

        c = TTLCache(ttl_seconds=60)
        c.set("x", 42)
        assert peek(c, "x") == 42

    def test_missing_key_returns_none(self) -> None:
        from app.cache import peek

        c = TTLCache(ttl_seconds=60)
        assert peek(c, "missing") is None

    def test_peek_does_not_change_hit_count(self) -> None:
        from app.cache import peek

        c = TTLCache(ttl_seconds=60)
        c.set("k", "v")
        peek(c, "k")
        assert c.hits == 0


class TestBatchDelete:
    def test_removes_existing_keys(self) -> None:
        from app.cache import batch_delete

        c = TTLCache(ttl_seconds=60)
        c.set("a", 1)
        c.set("b", 2)
        removed = batch_delete(c, ["a", "b"])
        assert removed == 2
        assert c.get("a") is None

    def test_nonexistent_keys_not_counted(self) -> None:
        from app.cache import batch_delete

        c = TTLCache(ttl_seconds=60)
        removed = batch_delete(c, ["x", "y"])
        assert removed == 0

    def test_partial_removal(self) -> None:
        from app.cache import batch_delete

        c = TTLCache(ttl_seconds=60)
        c.set("keep", 1)
        batch_delete(c, ["keep"])
        assert c.get("keep") is None

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_parametrized_batch_delete(self, n) -> None:
        from app.cache import batch_delete

        c = TTLCache(ttl_seconds=60)
        keys = [str(i) for i in range(n)]
        for k in keys:
            c.set(k, k)
        removed = batch_delete(c, keys)
        assert removed == n


class TestCacheKeyCount:
    def test_empty_cache(self) -> None:
        from app.cache import cache_key_count

        c = TTLCache(ttl_seconds=60)
        assert cache_key_count(c) == 0

    def test_after_inserts(self) -> None:
        from app.cache import cache_key_count

        c = TTLCache(ttl_seconds=60)
        c.set("a", 1)
        c.set("b", 2)
        assert cache_key_count(c) == 2


class TestWarmCacheNew:
    def test_loads_all_keys(self) -> None:
        from app.cache import warm_cache

        c = TTLCache(ttl_seconds=60)
        loaded = warm_cache(c, {"x": 1, "y": 2, "z": 3})
        assert loaded == 3
        assert c.get("x") == 1

    def test_empty_data(self) -> None:
        from app.cache import warm_cache

        c = TTLCache(ttl_seconds=60)
        assert warm_cache(c, {}) == 0


class TestGetOrDefault:
    def test_returns_cached_value(self) -> None:
        from app.cache import get_or_default

        c = TTLCache(ttl_seconds=60)
        c.set("k", "v")
        assert get_or_default(c, "k") == "v"

    def test_missing_key_returns_default(self) -> None:
        from app.cache import get_or_default

        c = TTLCache(ttl_seconds=60)
        assert get_or_default(c, "missing", default="fallback") == "fallback"

    def test_default_is_none_when_not_set(self) -> None:
        from app.cache import get_or_default

        c = TTLCache(ttl_seconds=60)
        assert get_or_default(c, "nope") is None


class TestCacheMissRate:
    def test_basic(self) -> None:
        from app.cache import cache_miss_rate
        assert cache_miss_rate(8, 2) == pytest.approx(0.2)

    def test_all_hits(self) -> None:
        from app.cache import cache_miss_rate
        assert cache_miss_rate(10, 0) == pytest.approx(0.0)

    def test_no_requests(self) -> None:
        from app.cache import cache_miss_rate
        assert cache_miss_rate(0, 0) == pytest.approx(0.0)

    def test_negative_raises(self) -> None:
        from app.cache import cache_miss_rate
        with pytest.raises(ValueError):
            cache_miss_rate(-1, 5)


class TestIsCacheEmpty:
    def test_empty_cache(self) -> None:
        from app.cache import is_cache_empty
        c = TTLCache(ttl_seconds=60)
        assert is_cache_empty(c) is True

    def test_non_empty_cache(self) -> None:
        from app.cache import is_cache_empty
        c = TTLCache(ttl_seconds=60)
        c.set("key", "val")
        assert is_cache_empty(c) is False


class TestCacheRemainingCapacity:
    def test_full_capacity_when_empty(self) -> None:
        from app.cache import cache_remaining_capacity
        c = TTLCache(ttl_seconds=60, max_size=5)
        assert cache_remaining_capacity(c) == 5

    def test_decreases_after_set(self) -> None:
        from app.cache import cache_remaining_capacity
        c = TTLCache(ttl_seconds=60, max_size=5)
        c.set("a", 1)
        assert cache_remaining_capacity(c) == 4
