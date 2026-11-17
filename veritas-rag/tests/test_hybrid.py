"""Tests for veritas-rag reciprocal-rank fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.hybrid import reciprocal_rank_fusion


def test_empty_inputs_return_empty():
    result = reciprocal_rank_fusion([], [])
    assert result == []


def test_bm25_only_results_returned():
    bm25 = [("c1", 0.9), ("c2", 0.7)]
    result = reciprocal_rank_fusion(bm25, [])
    ids = [r[0] for r in result]
    assert "c1" in ids and "c2" in ids


def test_vector_only_results_returned():
    vector = [("c1", 0.95), ("c3", 0.6)]
    result = reciprocal_rank_fusion([], vector)
    ids = [r[0] for r in result]
    assert "c1" in ids and "c3" in ids


def test_overlap_boosts_score():
    bm25 = [("shared", 0.9), ("bm25only", 0.5)]
    vector = [("shared", 0.95), ("veconly", 0.4)]
    result = reciprocal_rank_fusion(bm25, vector)
    result_map = {r[0]: r[1] for r in result}
    assert result_map["shared"] > result_map.get("bm25only", 0)
    assert result_map["shared"] > result_map.get("veconly", 0)


def test_result_sorted_descending():
    bm25 = [("a", 0.9), ("b", 0.5)]
    vector = [("b", 0.8), ("c", 0.3)]
    result = reciprocal_rank_fusion(bm25, vector)
    scores = [r[1] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_result_tuples_have_four_elements():
    bm25 = [("c1", 0.5)]
    vector = [("c2", 0.4)]
    result = reciprocal_rank_fusion(bm25, vector)
    for item in result:
        assert len(item) == 4


def test_bm25_rank_recorded_correctly():
    bm25 = [("c1", 0.9), ("c2", 0.5)]
    vector = []
    result = reciprocal_rank_fusion(bm25, vector)
    result_map = {r[0]: r for r in result}
    assert result_map["c1"][2] == 1  # bm25 rank 1
    assert result_map["c2"][2] == 2  # bm25 rank 2


def test_vector_rank_recorded_correctly():
    bm25 = []
    vector = [("x", 0.9), ("y", 0.4)]
    result = reciprocal_rank_fusion(bm25, vector)
    result_map = {r[0]: r for r in result}
    assert result_map["x"][3] == 1
    assert result_map["y"][3] == 2


def test_missing_bm25_rank_is_none():
    bm25 = [("a", 0.9)]
    vector = [("b", 0.8)]
    result = reciprocal_rank_fusion(bm25, vector)
    result_map = {r[0]: r for r in result}
    assert result_map["b"][2] is None  # b not in bm25


@pytest.mark.parametrize("k", [10, 60, 100])
def test_custom_k_produces_valid_results(k):
    bm25 = [("c1", 0.9), ("c2", 0.5)]
    vector = [("c1", 0.8)]
    result = reciprocal_rank_fusion(bm25, vector, k=k)
    assert len(result) >= 1
