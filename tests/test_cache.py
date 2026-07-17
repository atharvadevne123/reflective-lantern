"""TTL cache tests."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache


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


def test_contains_present_key():
    c = TTLCache(ttl_seconds=10)
    c.set("x", 99)
    assert "x" in c


def test_contains_missing_key():
    c = TTLCache(ttl_seconds=10)
    assert "missing" not in c


def test_stats_returns_expected_keys():
    c = TTLCache(ttl_seconds=10, max_size=50)
    c.set("a", 1)
    _ = c.get("a")
    _ = c.get("b")
    stats = c.stats()
    assert stats["size"] == 1
    assert stats["max_size"] == 50
    assert stats["ttl_seconds"] == 10
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


def test_stats_empty_cache():
    c = TTLCache()
    stats = c.stats()
    assert stats["size"] == 0
    assert stats["hit_rate"] == 0.0


def test_eviction_count_initial_zero():
    c = TTLCache(ttl_seconds=60, max_size=10)
    assert c.eviction_count == 0


def test_eviction_count_increments_on_overflow():
    c = TTLCache(ttl_seconds=60, max_size=3)
    for i in range(5):
        c.set(f"key_{i}", i)
    assert c.eviction_count == 2


def test_eviction_count_resets_on_clear():
    c = TTLCache(ttl_seconds=60, max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # triggers eviction
    assert c.eviction_count == 1
    c.clear()
    assert c.eviction_count == 0


def test_stats_includes_evictions():
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
def test_eviction_count_parametrized(max_size, n_inserts, expected_evictions):
    c = TTLCache(ttl_seconds=60, max_size=max_size)
    for i in range(n_inserts):
        c.set(f"k_{i}", i)
    assert c.eviction_count == expected_evictions


def test_warm_cache_inserts_entries():
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    count = warm_cache(c, {"a": 1, "b": 2, "c": 3})
    assert count == 3
    assert c.get("a") == 1
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_warm_cache_empty_dict():
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    count = warm_cache(c, {})
    assert count == 0


def test_warm_cache_returns_insert_count():
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    entries = {f"key_{i}": i for i in range(10)}
    count = warm_cache(c, entries)
    assert count == 10


def test_warm_cache_overwrite_existing():
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    c.set("x", "old")
    warm_cache(c, {"x": "new"})
    assert c.get("x") == "new"


@pytest.mark.parametrize("n", [1, 5, 20])
def test_warm_cache_parametrized_count(n):
    from app.cache import TTLCache, warm_cache

    c = TTLCache(ttl_seconds=60, max_size=100)
    entries = {str(i): i * 10 for i in range(n)}
    assert warm_cache(c, entries) == n
