"""FAISS/numpy retriever tests."""
from __future__ import annotations

import numpy as np
import pytest

from app.retriever import TicketRetriever


def test_retriever_empty_returns_nothing():
    r = TicketRetriever(dim=8)
    result = r.search(np.zeros(8), k=3)
    assert result == []


def test_retriever_add_and_search():
    r = TicketRetriever(dim=8)
    v1 = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    r.add(v1, {"id": "t1", "category": "network"})
    r.add(v2, {"id": "t2", "category": "access"})

    results = r.search(v1, k=1)
    assert len(results) == 1
    assert results[0]["id"] == "t1"


def test_retriever_size():
    r = TicketRetriever(dim=4)
    assert r.size == 0
    r.add(np.ones(4), {"id": "x"})
    assert r.size == 1


@pytest.mark.parametrize("k", [1, 3, 5])
def test_retriever_k_results(k):
    r = TicketRetriever(dim=4)
    for i in range(10):
        r.add(np.random.rand(4), {"id": f"t{i}"})
    results = r.search(np.random.rand(4), k=k)
    assert len(results) == k
