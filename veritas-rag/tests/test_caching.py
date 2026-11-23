"""Tests for veritas-rag app/caching.py."""

from __future__ import annotations

import time

from app.caching import (
    RetrievalCache,
    cache_age_seconds,
    cache_fill_ratio,
    cache_key,
    normalize_query,
    top_cached_queries,
)


class TestNormalizeQuery:
    def test_lowercases(self) -> None:
        assert normalize_query("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self) -> None:
        result = normalize_query("hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_single_spaces(self) -> None:
        result = normalize_query("hello   world")
        assert "  " not in result

    def test_empty_string(self) -> None:
        assert normalize_query("") == ""

    def test_preserves_words(self) -> None:
        result = normalize_query("what is the capital")
        assert "what" in result
        assert "capital" in result


class TestCacheKey:
    def test_returns_string(self) -> None:
        key = cache_key("test query")
        assert isinstance(key, str)

    def test_fixed_length(self) -> None:
        key = cache_key("any query")
        assert len(key) == 32

    def test_deterministic(self) -> None:
        assert cache_key("same") == cache_key("same")

    def test_different_queries_different_keys(self) -> None:
        assert cache_key("query a") != cache_key("query b")


class TestRetrievalCache:
    def test_miss_on_empty(self) -> None:
        cache = RetrievalCache()
        assert cache.get("anything") is None

    def test_put_and_get(self) -> None:
        cache = RetrievalCache()
        result = {"chunks": ["a"]}
        cache.put("query", result)
        assert cache.get("query") == result

    def test_hit_increments_counter(self) -> None:
        cache = RetrievalCache()
        cache.put("q", {"data": 1})
        before_hits = cache.hits
        cache.get("q")
        assert cache.hits > before_hits

    def test_miss_increments_counter(self) -> None:
        cache = RetrievalCache()
        before_misses = cache.misses
        cache.get("not_cached")
        assert cache.misses > before_misses

    def test_evicts_oldest_when_full(self) -> None:
        cache = RetrievalCache(max_size=3)
        for i in range(4):
            cache.put(f"q{i}", {"i": i})
        assert len(cache) <= 3

    def test_ttl_expiry(self) -> None:
        cache = RetrievalCache(ttl_seconds=0.05)
        cache.put("q", {"data": 1})
        time.sleep(0.1)
        assert cache.get("q") is None

    def test_stats_returns_dict(self) -> None:
        cache = RetrievalCache()
        stats = cache.stats()
        assert isinstance(stats, dict)

    def test_invalidate_all_clears_cache(self) -> None:
        cache = RetrievalCache()
        for i in range(5):
            cache.put(f"query{i}", {"data": i})
        cache.invalidate_all()
        assert len(cache) == 0

    def test_warm_populates_cache(self) -> None:
        cache = RetrievalCache()
        pairs = [("query one", {"r": 1}), ("query two", {"r": 2})]
        inserted = cache.warm(pairs)
        assert inserted == 2
        assert cache.get("query one") is not None

    def test_stats_hit_rate_increases_on_hits(self) -> None:
        cache = RetrievalCache()
        cache.put("q", {"data": 1})
        cache.get("q")
        cache.get("q")
        stats = cache.stats()
        assert stats["hits"] == 2

    def test_cache_len_reflects_entries(self) -> None:
        cache = RetrievalCache()
        cache.put("a", 1)
        cache.put("b", 2)
        assert len(cache) == 2

    def test_put_and_get_same_value(self) -> None:
        cache = RetrievalCache()
        cache.put("my query", {"result": "answer"})
        result = cache.get("my query")
        assert result == {"result": "answer"}

    def test_evict_expired_removes_entries(self) -> None:
        cache = RetrievalCache(ttl_seconds=0.05)
        cache.put("q1", {"data": 1})
        cache.put("q2", {"data": 2})
        time.sleep(0.1)
        evicted = cache.evict_expired()
        assert evicted == 2
        assert len(cache) == 0

    def test_evict_expired_keeps_live_entries(self) -> None:
        cache = RetrievalCache(ttl_seconds=10.0)
        cache.put("q", {"data": 1})
        evicted = cache.evict_expired()
        assert evicted == 0
        assert len(cache) == 1


class TestCacheUtilities:
    def test_cache_fill_ratio_empty(self) -> None:
        cache = RetrievalCache(max_size=100)
        assert cache_fill_ratio(cache) == 0.0

    def test_cache_fill_ratio_half(self) -> None:
        cache = RetrievalCache(max_size=4)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache_fill_ratio(cache) == 0.5

    def test_cache_fill_ratio_full(self) -> None:
        cache = RetrievalCache(max_size=2)
        cache.put("x", 1)
        cache.put("y", 2)
        assert cache_fill_ratio(cache) == 1.0

    def test_top_cached_queries_order(self) -> None:
        cache = RetrievalCache()
        for q in ["first", "second", "third"]:
            cache.put(q, q)
        top = top_cached_queries(cache, n=2)
        assert len(top) == 2

    def test_top_cached_queries_empty(self) -> None:
        cache = RetrievalCache()
        assert top_cached_queries(cache, n=5) == []

    def test_cache_age_seconds_present(self) -> None:
        cache = RetrievalCache()
        cache.put("q", {"data": 1})
        age = cache_age_seconds(cache, "q")
        assert age is not None
        assert age >= 0.0

    def test_cache_age_seconds_absent(self) -> None:
        cache = RetrievalCache()
        assert cache_age_seconds(cache, "not_there") is None

    def test_cache_age_seconds_expired(self) -> None:
        cache = RetrievalCache(ttl_seconds=0.05)
        cache.put("q", {"data": 1})
        time.sleep(0.1)
        assert cache_age_seconds(cache, "q") is None
