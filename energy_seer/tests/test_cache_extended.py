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


class TestGetOrSetAndUpdateMany:
    def setup_method(self) -> None:
        from app.cache import clear_all
        clear_all()

    def test_get_or_set_returns_existing_value(self) -> None:
        from app.cache import get_or_set, set_cache
        set_cache("k", "original")
        result = get_or_set("k", "new_value")
        assert result == "original"

    def test_get_or_set_sets_missing_key(self) -> None:
        from app.cache import get_cache, get_or_set
        result = get_or_set("missing", "default_val")
        assert result == "default_val"
        assert get_cache("missing") == "default_val"

    def test_update_many_sets_all_entries(self) -> None:
        from app.cache import get_cache, update_many
        update_many({"a": 1, "b": 2, "c": 3})
        assert get_cache("a") == 1
        assert get_cache("b") == 2
        assert get_cache("c") == 3

    def test_update_many_empty_dict_no_error(self) -> None:
        from app.cache import update_many
        update_many({})  # should not raise


class TestCacheDefaultTTLExtended:
    def setup_method(self) -> None:
        from app.cache import clear_all
        clear_all()

    def test_set_and_get_string_value(self) -> None:
        from app.cache import get_cache, set_cache
        set_cache("greeting", "hello")
        assert get_cache("greeting") == "hello"

    def test_set_and_get_list_value(self) -> None:
        from app.cache import get_cache, set_cache
        set_cache("nums", [1, 2, 3])
        assert get_cache("nums") == [1, 2, 3]

    def test_set_and_get_dict_value(self) -> None:
        from app.cache import get_cache, set_cache
        set_cache("data", {"k": "v"})
        assert get_cache("data") == {"k": "v"}

    def test_overwrite_existing_key(self) -> None:
        from app.cache import get_cache, set_cache
        set_cache("x", 1)
        set_cache("x", 2)
        assert get_cache("x") == 2

    def test_invalidate_specific_key(self) -> None:
        from app.cache import get_cache, invalidate, set_cache
        set_cache("a", 1)
        set_cache("b", 2)
        invalidate("a")
        assert get_cache("a") is None
        assert get_cache("b") == 2

    @pytest.mark.parametrize("key", ["k1", "k2", "k3", "k4", "k5"])
    def test_various_keys_independent(self, key: str) -> None:
        from app.cache import get_cache, set_cache
        set_cache(key, key + "_val")
        assert get_cache(key) == key + "_val"
