"""Extended tests for energy_seer app/cache.py TTL cache."""

from __future__ import annotations

import time

import pytest

from app.cache import clear_all, get_cache, set_cache


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
