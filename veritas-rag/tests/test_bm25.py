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


def test_bm25_stopword_only_query_returns_empty():
    idx = BM25Index()
    idx.add("c1", "some relevant content about machine learning")
    results = idx.search("the a an is")
    assert results == []


def test_bm25_scores_are_positive():
    idx = BM25Index()
    idx.add("c1", "python machine learning neural network")
    results = idx.search("machine learning", top_k=5)
    assert all(score >= 0 for _, score in results)


def test_bm25_top_k_limits_results():
    idx = BM25Index()
    for i in range(10):
        idx.add(f"c{i}", f"python machine learning document number {i}")
    results = idx.search("python machine", top_k=3)
    assert len(results) <= 3


def test_bm25_remove_nonexistent_no_error():
    idx = BM25Index()
    idx.remove("nonexistent-chunk")
    assert len(idx) == 0


def test_bm25_relevance_ordering():
    idx = BM25Index()
    idx.add("relevant", "python machine learning neural network deep learning")
    idx.add("irrelevant", "cooking pasta recipe italian food")
    results = idx.search("machine learning python", top_k=2)
    assert results[0][0] == "relevant"


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (2.0, 0.5), (1.5, 0.0)])
def test_bm25_custom_params(k1, b):
    idx = BM25Index(k1=k1, b=b)
    idx.add("c1", "python machine learning")
    results = idx.search("python")
    assert len(results) == 1


def test_bm25_scores_sorted_descending():
    idx = BM25Index()
    idx.add("c1", "machine learning neural network deep learning architecture")
    idx.add("c2", "machine learning basics tutorial")
    idx.add("c3", "cooking pasta recipe")
    results = idx.search("machine learning neural", top_k=3)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_exact_match_higher_than_partial():
    idx = BM25Index()
    idx.add("exact", "neural network architecture design patterns")
    idx.add("partial", "some unrelated text with neural")
    results = idx.search("neural network architecture", top_k=2)
    result_map = {cid: s for cid, s in results}
    assert result_map.get("exact", 0) >= result_map.get("partial", 0)


def test_bm25_update_by_remove_and_add():
    idx = BM25Index()
    idx.add("c1", "original content about machine learning")
    idx.remove("c1")
    idx.add("c1", "updated content about deep learning transformers")
    results = idx.search("transformers deep", top_k=1)
    assert len(results) == 1


def test_bm25_multiple_removes_do_not_error():
    idx = BM25Index()
    idx.add("c1", "some content")
    idx.remove("c1")
    idx.remove("c1")  # second remove should be a no-op
    assert len(idx) == 0
