"""Tests for veritas-rag app/retrieval/bm25.py."""

from __future__ import annotations

import pytest

from app.retrieval.bm25 import BM25Index, tokenize


class TestTokenize:
    def test_lowercases(self) -> None:
        tokens = tokenize("HELLO WORLD")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_stopwords(self) -> None:
        tokens = tokenize("the quick fox")
        assert "the" not in tokens

    def test_removes_punctuation(self) -> None:
        tokens = tokenize("hello, world!")
        assert "," not in tokens

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_alphanumeric_kept(self) -> None:
        tokens = tokenize("model v2 test")
        assert "model" in tokens
        assert "v2" in tokens


class TestBM25Index:
    def _build(self) -> BM25Index:
        idx = BM25Index()
        idx.add("c1", "Paris is the capital of France")
        idx.add("c2", "Berlin is the capital of Germany")
        idx.add("c3", "Tokyo is the capital of Japan")
        return idx

    def test_len_after_add(self) -> None:
        idx = self._build()
        assert len(idx) == 3

    def test_search_returns_list(self) -> None:
        idx = self._build()
        results = idx.search("Paris France", top_k=5)
        assert isinstance(results, list)

    def test_search_returns_tuples(self) -> None:
        idx = self._build()
        results = idx.search("capital", top_k=3)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_relevant_result_ranked_first(self) -> None:
        idx = self._build()
        results = idx.search("Paris France", top_k=3)
        assert results[0][0] == "c1"

    def test_top_k_limits_results(self) -> None:
        idx = self._build()
        results = idx.search("capital", top_k=2)
        assert len(results) <= 2

    def test_remove_decrements_len(self) -> None:
        idx = self._build()
        idx.remove("c1")
        assert len(idx) == 2

    def test_remove_nonexistent_no_error(self) -> None:
        idx = self._build()
        idx.remove("does_not_exist")

    def test_empty_query_returns_empty(self) -> None:
        idx = self._build()
        results = idx.search("", top_k=5)
        assert results == []

    def test_empty_index_returns_empty(self) -> None:
        idx = BM25Index()
        results = idx.search("test query", top_k=5)
        assert results == []
