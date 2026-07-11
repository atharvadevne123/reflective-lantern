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


def test_missing_key_returns_none(cache) -> None:
    assert cache.get("nonexistent") is None


def test_expired_entry_returns_none(cache) -> None:
    tiny = TTLCache(ttl_seconds=0.01)
    tiny.set("k", "v")
    time.sleep(0.02)
    assert tiny.get("k") is None


def test_len_counts_entries(cache) -> None:
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2


def test_invalidate_removes_key(cache) -> None:
    cache.set("k", "v")
    existed = cache.invalidate("k")
    assert existed is True
    assert cache.get("k") is None


def test_invalidate_missing_key_returns_false(cache) -> None:
    assert cache.invalidate("missing") is False


def test_clear_removes_all(cache) -> None:
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0


def test_max_size_evicts_oldest(cache) -> None:
    for i in range(5):
        cache.set(f"k{i}", i)
    assert len(cache) == 5
    cache.set("k5", 5)
    assert len(cache) == 5


@pytest.mark.parametrize("value", [None, 0, "", [], {}, False])
def test_falsy_values_stored_and_retrieved(cache, value) -> None:
    cache.set("falsy", value)
    assert cache.get("falsy") == value


def test_overwrite_resets_ttl(cache) -> None:
    cache.set("k", "v1")
    cache.set("k", "v2")
    assert cache.get("k") == "v2"


def test_multiple_keys_independent(cache) -> None:
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2


@pytest.mark.parametrize("key", ["key1", "key-two", "KEY_THREE", "k123"])
def test_various_key_formats(cache, key) -> None:
    cache.set(key, "value")
    assert cache.get(key) == "value"


def test_clear_then_set(cache) -> None:
    cache.set("x", 10)
    cache.clear()
    cache.set("x", 20)
    assert cache.get("x") == 20


def test_len_after_expiry(cache) -> None:
    tiny = TTLCache(ttl_seconds=0.01, max_size=10)
    tiny.set("a", 1)
    time.sleep(0.02)
    tiny.get("a")
    assert len(tiny) == 0


@pytest.mark.parametrize("ttl", [0.1, 1.0, 10.0])
def test_cache_respects_different_ttls(ttl) -> None:
    c = TTLCache(ttl_seconds=ttl)
    c.set("k", "v")
    assert c.get("k") == "v"


def test_invalidate_then_set(cache) -> None:
    cache.set("k", "old")
    cache.invalidate("k")
    cache.set("k", "new")
    assert cache.get("k") == "new"


def test_keys_returns_stored_keys(cache) -> None:
    cache.set("alpha", 1)
    cache.set("beta", 2)
    keys = cache.keys()
    assert "alpha" in keys
    assert "beta" in keys


def test_keys_empty_on_fresh_cache() -> None:
    c = TTLCache()
    assert c.keys() == []


def test_ttl_property_returns_configured_value() -> None:
    c = TTLCache(ttl_seconds=120.0)
    assert c.ttl == 120.0


def test_max_size_property_returns_configured_value() -> None:
    c = TTLCache(max_size=64)
    assert c.max_size == 64


@pytest.mark.parametrize("ttl", [30.0, 120.0, 600.0])
def test_ttl_property_parametrized(ttl: float) -> None:
    c = TTLCache(ttl_seconds=ttl)
    assert c.ttl == ttl


def test_keys_after_invalidate_removes_key(cache) -> None:
    cache.set("to_remove", "x")
    cache.set("keep", "y")
    cache.invalidate("to_remove")
    assert "to_remove" not in cache.keys()
    assert "keep" in cache.keys()


def test_neighborhood_cache_has_positive_ttl() -> None:
    from app.cache import neighborhood_cache
    assert neighborhood_cache.ttl > 0


def test_metrics_cache_has_positive_ttl() -> None:
    from app.cache import metrics_cache
    assert metrics_cache.ttl > 0


def test_max_size_property() -> None:
    cache = TTLCache(max_size=10, ttl_seconds=60)
    assert cache.max_size == 10


def test_max_size_property_reflects_constructor_arg() -> None:
    for n in [5, 50, 500]:
        cache = TTLCache(max_size=n, ttl_seconds=60)
        assert cache.max_size == n


def test_keys_returns_empty_list_on_new_cache() -> None:
    cache = TTLCache(max_size=10, ttl_seconds=60)
    assert cache.keys() == []


def test_keys_contains_set_key() -> None:
    cache = TTLCache(max_size=10, ttl_seconds=60)
    cache.set("alpha", 1)
    assert "alpha" in cache.keys()


def test_keys_reflects_all_stored_keys() -> None:
    cache = TTLCache(max_size=10, ttl_seconds=60)
    for k in ("a", "b", "c"):
        cache.set(k, k)
    assert set(cache.keys()) == {"a", "b", "c"}


@pytest.mark.parametrize("ttl", [30, 60, 300, 3600])
def test_ttl_property_various_values(ttl: int) -> None:
    cache = TTLCache(max_size=10, ttl_seconds=ttl)
    assert cache.ttl == ttl
