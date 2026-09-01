"""Tests for app/utils/cache.py."""

from __future__ import annotations

import time

import pytest


def test_cache_set_get():
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=60)
    c.set("k", 42)
    assert c.get("k") == 42


def test_cache_miss_returns_none():
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=60)
    assert c.get("missing") is None


def test_cache_delete():
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=60)
    c.set("k", 1)
    c.delete("k")
    assert c.get("k") is None


def test_cache_clear():
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert len(c) == 0


def test_cache_expired():
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=0)
    c.set("k", 99)
    time.sleep(0.01)
    assert c.get("k") is None


def test_cache_delete_nonexistent_returns_false() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache()
    assert c.delete("nonexistent") is False


def test_cache_max_size_evicts_oldest() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=300, max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert len(c) == 2


def test_cache_set_overwrites_existing() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache()
    c.set("k", 1)
    c.set("k", 2)
    assert c.get("k") == 2


def test_cache_size_matches_len() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache()
    c.set("a", 1)
    c.set("b", 2)
    assert c.size() == len(c) == 2


def test_cache_items_returns_all_valid() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=300)
    c.set("x", 10)
    c.set("y", 20)
    pairs = c.items()
    keys = [k for k, _ in pairs]
    assert "x" in keys and "y" in keys


def test_cache_items_excludes_expired() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=0)
    c.set("x", 10)
    time.sleep(0.01)
    pairs = c.items()
    assert pairs == []


def test_cache_estimated_valid_count() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=300)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert c.estimated_valid_count() == 3


def test_cache_estimated_valid_count_zero_after_clear() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=300)
    c.set("a", 1)
    c.clear()
    assert c.estimated_valid_count() == 0


@pytest.mark.parametrize("key,value", [("str", "hello"), ("int", 42), ("list", [1, 2, 3])])
def test_cache_set_and_get_various_types(key: str, value) -> None:
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=300)
    c.set(key, value)
    assert c.get(key) == value


def test_cache_overwrite_key_returns_new_value() -> None:
    from app.utils.cache import TTLCache

    c = TTLCache(ttl_seconds=300)
    c.set("k", 1)
    c.set("k", 99)
    assert c.get("k") == 99
