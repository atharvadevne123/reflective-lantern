"""Tests for in-memory TTL cache."""

import time

import pytest
from app.cache import TTLCache


@pytest.fixture
def cache():
    return TTLCache(ttl_seconds=1.0, max_size=5)


def test_set_and_get(cache):
    cache.set("k1", "value1")
    assert cache.get("k1") == "value1"


def test_missing_key_returns_none(cache):
    assert cache.get("nonexistent") is None


def test_expired_entry_returns_none(cache):
    tiny = TTLCache(ttl_seconds=0.01)
    tiny.set("k", "v")
    time.sleep(0.02)
    assert tiny.get("k") is None


def test_len_counts_entries(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2


def test_invalidate_removes_key(cache):
    cache.set("k", "v")
    existed = cache.invalidate("k")
    assert existed is True
    assert cache.get("k") is None


def test_invalidate_missing_key_returns_false(cache):
    assert cache.invalidate("missing") is False


def test_clear_removes_all(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0


def test_max_size_evicts_oldest(cache):
    for i in range(5):
        cache.set(f"k{i}", i)
    assert len(cache) == 5
    cache.set("k5", 5)
    assert len(cache) == 5


@pytest.mark.parametrize("value", [None, 0, "", [], {}, False])
def test_falsy_values_stored_and_retrieved(cache, value):
    cache.set("falsy", value)
    assert cache.get("falsy") == value


def test_overwrite_resets_ttl(cache):
    cache.set("k", "v1")
    cache.set("k", "v2")
    assert cache.get("k") == "v2"
