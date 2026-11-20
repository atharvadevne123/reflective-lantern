"""Ticket similarity search tests for Ticket-Oracle."""

from __future__ import annotations

import numpy as np
import pytest

from app.similarity import build_index, is_index_built, search_similar


@pytest.fixture(autouse=True)
def reset_index():
    """Reset similarity module state between tests."""
    import app.similarity as sim
    sim._index = None
    sim._metadata = []
    yield
    sim._index = None
    sim._metadata = []


def _make_vectors(n: int = 20, d: int = 16) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(42)
    vecs = rng.random((n, d)).astype(np.float32)
    meta = [{"ticket_id": f"T-{i}", "priority": "P2"} for i in range(n)]
    return vecs, meta


def test_build_index_sets_index_flag():
    vecs, meta = _make_vectors()
    build_index(vecs, meta)
    assert is_index_built()


def test_search_returns_top_k():
    vecs, meta = _make_vectors(20)
    build_index(vecs, meta)
    query = vecs[0]
    results = search_similar(query, top_k=3)
    assert len(results) == 3


def test_search_first_result_is_closest():
    vecs, meta = _make_vectors(20)
    build_index(vecs, meta)
    # Query with vecs[5]: it should appear in the top-1 results
    results = search_similar(vecs[5], top_k=1)
    assert results[0]["ticket_id"] == "T-5"


def test_search_distance_is_non_negative():
    vecs, meta = _make_vectors()
    build_index(vecs, meta)
    results = search_similar(vecs[0], top_k=5)
    assert all(r["distance"] >= 0 for r in results)


def test_search_metadata_preserved():
    vecs, meta = _make_vectors(10)
    build_index(vecs, meta)
    results = search_similar(vecs[0], top_k=3)
    for r in results:
        assert "ticket_id" in r
        assert "priority" in r


def test_search_before_build_raises():
    with pytest.raises(RuntimeError, match="not built"):
        search_similar(np.zeros(16, dtype=np.float32))


def test_is_index_built_false_initially():
    assert not is_index_built()


def test_build_index_accepts_1d_query():
    vecs, meta = _make_vectors()
    build_index(vecs, meta)
    query_1d = vecs[0]
    results = search_similar(query_1d, top_k=2)
    assert len(results) == 2


def test_top_k_clamped_to_index_size():
    vecs, meta = _make_vectors(5)
    build_index(vecs, meta)
    results = search_similar(vecs[0], top_k=100)
    assert len(results) == 5
