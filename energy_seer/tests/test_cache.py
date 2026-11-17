"""Tests for Energy-Seer in-memory TTL cache."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.cache as cache_module
from app.cache import (
    cache_size,
    clear_all,
    evict_expired,
    get_cache,
    invalidate,
    set_cache,
)


@pytest.fixture(autouse=True)
def reset_cache():
    cache_module._store.clear()
    yield
    cache_module._store.clear()


def test_set_and_get():
    set_cache("k1", {"result": 42})
    value = get_cache("k1")
    assert value == {"result": 42}


def test_missing_key_returns_none():
    assert get_cache("missing") is None


def test_expired_key_returns_none():
    set_cache("exp", "val", ttl=0.01)
    time.sleep(0.05)
    assert get_cache("exp") is None


def test_non_expired_key_returns_value():
    set_cache("fresh", "data", ttl=100.0)
    assert get_cache("fresh") == "data"


def test_invalidate_removes_key():
    set_cache("k2", "value")
    invalidate("k2")
    assert get_cache("k2") is None


def test_invalidate_missing_key_no_error():
    invalidate("does_not_exist")


def test_clear_all_empties_store():
    set_cache("a", 1)
    set_cache("b", 2)
    clear_all()
    assert get_cache("a") is None
    assert get_cache("b") is None


def test_cache_size_counts_non_expired():
    set_cache("x", 1, ttl=100.0)
    set_cache("y", 2, ttl=100.0)
    assert cache_size() == 2


def test_evict_expired_removes_stale():
    set_cache("stale", "v", ttl=0.01)
    set_cache("fresh", "v2", ttl=100.0)
    time.sleep(0.05)
    count = evict_expired()
    assert count >= 1
    assert get_cache("fresh") == "v2"


def test_overwrite_existing_key():
    set_cache("k", "old")
    set_cache("k", "new")
    assert get_cache("k") == "new"


@pytest.mark.parametrize("value", [42, "string", [1, 2], {"a": 1}])
def test_cache_various_value_types(value):
    set_cache("typed_key", value)
    assert get_cache("typed_key") == value
