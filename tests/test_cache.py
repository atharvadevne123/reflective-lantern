"""Tests for in-memory TTL cache."""

import time

import pytest

from app.cache import TTLCache


@pytest.fixture
def cache() -> None:
    return TTLCache(ttl_seconds=1.0, max_size=5)


def test_set_and_get(cache) -> None:
    cache.set("k1", "value1")
    assert cache.get("k1") == "value1"


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
