"""Extended tests for energy_seer app/cache.py TTL cache."""

from __future__ import annotations

import time

import pytest

from app.cache import (
    cache_size,
    clear_all,
    evict_expired,
    get_cache,
    get_with_default,
    invalidate,
    keys,
    set_cache,
)


def _reset() -> None:
    clear_all()


class TestSetAndGetCacheExtended:
    def setup_method(self) -> None:
        _reset()

    def test_set_then_get_returns_value(self) -> None:
        set_cache("k", 42)
        assert get_cache("k") == 42

    def test_missing_key_returns_none(self) -> None:
        assert get_cache("nonexistent") is None

    def test_expired_entry_returns_none(self) -> None:
        set_cache("k", "v", ttl=0.01)
        time.sleep(0.05)
        assert get_cache("k") is None

    def test_overwrite_existing_key(self) -> None:
        set_cache("k", "first")
        set_cache("k", "second")
        assert get_cache("k") == "second"

    def test_stores_dict_value(self) -> None:
        set_cache("config", {"a": 1, "b": 2})
        assert get_cache("config") == {"a": 1, "b": 2}

    def test_stores_list_value(self) -> None:
        set_cache("lst", [1, 2, 3])
        assert get_cache("lst") == [1, 2, 3]

    def test_stores_none_value(self) -> None:
        set_cache("n", None)
        assert get_cache("n") is None

    @pytest.mark.parametrize("value", [0, 0.0, "", [], {}, False])
    def test_falsy_values_stored_and_retrieved(self, value: object) -> None:
        set_cache("falsy", value)
        assert get_cache("falsy") == value


class TestInvalidateAndClearExtended:
    def setup_method(self) -> None:
        _reset()

    def test_invalidate_removes_key(self) -> None:
        set_cache("x", 1)
        invalidate("x")
        assert get_cache("x") is None

    def test_invalidate_nonexistent_no_error(self) -> None:
        invalidate("does_not_exist")  # should not raise

    def test_clear_all_removes_all(self) -> None:
        set_cache("a", 1)
        set_cache("b", 2)
        clear_all()
        assert get_cache("a") is None
        assert get_cache("b") is None

    def test_invalidate_does_not_affect_other_keys(self) -> None:
        set_cache("keep", "yes")
        set_cache("remove", "bye")
        invalidate("remove")
        assert get_cache("keep") == "yes"
        assert get_cache("remove") is None


class TestCacheSizeAndEvictExtended:
    def setup_method(self) -> None:
        _reset()

    def test_empty_cache_size_zero(self) -> None:
        assert cache_size() == 0

    def test_size_increases_on_add(self) -> None:
        set_cache("a", 1)
        set_cache("b", 2)
        assert cache_size() == 2

    def test_expired_entries_not_counted(self) -> None:
        set_cache("old", "v", ttl=0.01)
        set_cache("new", "v", ttl=60.0)
        time.sleep(0.05)
        assert cache_size() == 1

    def test_evict_expired_returns_count(self) -> None:
        set_cache("exp1", 1, ttl=0.01)
        set_cache("exp2", 2, ttl=0.01)
        set_cache("live", 3, ttl=60.0)
        time.sleep(0.05)
        removed = evict_expired()
        assert removed == 2

    def test_evict_expired_leaves_live_entries(self) -> None:
        set_cache("live", "here", ttl=60.0)
        set_cache("dead", "gone", ttl=0.01)
        time.sleep(0.05)
        evict_expired()
        assert get_cache("live") == "here"


class TestGetWithDefaultAndKeysExtended:
    def setup_method(self) -> None:
        _reset()

    def test_get_with_default_returns_value(self) -> None:
        set_cache("k", 99)
        assert get_with_default("k", default=0) == 99

    def test_get_with_default_missing_key(self) -> None:
        assert get_with_default("missing", default="fallback") == "fallback"

    def test_get_with_default_expired_returns_default(self) -> None:
        set_cache("x", "v", ttl=0.01)
        time.sleep(0.05)
        assert get_with_default("x", default="default") == "default"

    def test_keys_returns_live_keys(self) -> None:
        set_cache("alpha", 1)
        set_cache("beta", 2)
        live = keys()
        assert "alpha" in live
        assert "beta" in live

    def test_keys_excludes_expired(self) -> None:
        set_cache("alive", 1, ttl=60.0)
        set_cache("dead", 2, ttl=0.01)
        time.sleep(0.05)
        live = keys()
        assert "alive" in live
        assert "dead" not in live

    def test_keys_empty_after_clear(self) -> None:
        set_cache("x", 1)
        clear_all()
        assert keys() == []
