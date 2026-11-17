"""Tests for Sports-Oracle FAISS index with brute-force fallback."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.faiss_index import build_index, search_similar


def _embeddings(n: int = 5, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim)).astype(np.float32)


def test_build_index_without_faiss_returns_none():
    with patch.dict("sys.modules", {"faiss": None}):
        result = build_index(_embeddings())
    assert result is None


def test_search_similar_fallback_no_embeddings_returns_empty():
    result = search_similar(None, _embeddings(1)[0])
    assert result == []


def test_search_similar_fallback_with_embeddings():
    embs = _embeddings()
    query = embs[0]
    result = search_similar(None, query, k=3, embeddings=embs)
    assert isinstance(result, list)
    assert len(result) <= 3


def test_search_similar_fallback_returns_int_indices():
    embs = _embeddings()
    query = embs[0]
    result = search_similar(None, query, k=2, embeddings=embs)
    for idx in result:
        assert isinstance(idx, (int, np.integer))


def test_search_similar_fallback_top_k_closest():
    embs = _embeddings(10)
    query = embs[2].copy()
    result = search_similar(None, query, k=1, embeddings=embs)
    assert len(result) == 1
    assert result[0] == 2


def test_search_similar_fallback_respects_k():
    embs = _embeddings(10)
    for k in (1, 3, 5):
        result = search_similar(None, embs[0], k=k, embeddings=embs)
        assert len(result) == k


def test_build_index_with_mock_faiss():
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_faiss.IndexFlatL2.return_value = mock_index
    with patch.dict("sys.modules", {"faiss": mock_faiss}):
        result = build_index(_embeddings())
    assert result is mock_index


def test_search_similar_with_mock_faiss_index():
    embs = _embeddings()
    query = embs[0]
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
    result = search_similar(mock_index, query, k=2, embeddings=embs)
    assert isinstance(result, list)


@pytest.mark.parametrize("n,k", [(5, 3), (10, 5), (3, 2)])
def test_search_similar_fallback_various_sizes(n, k):
    embs = _embeddings(n)
    result = search_similar(None, embs[0], k=k, embeddings=embs)
    assert len(result) <= k
