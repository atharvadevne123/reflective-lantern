"""Extended tests for FAISS index utilities including size and clear."""

from __future__ import annotations

import numpy as np
import pytest

from app.faiss_index import clear_index, index_size


@pytest.fixture(autouse=True)
def reset_index():
    clear_index()
    yield
    clear_index()


class TestIndexSize:
    def test_empty_index_returns_zero(self) -> None:
        assert index_size() == 0

    def test_returns_int(self) -> None:
        assert isinstance(index_size(), int)

    def test_size_after_build(self) -> None:
        from app.faiss_index import build_index
        vectors = np.random.default_rng(0).standard_normal((10, 5)).astype(np.float32)
        labels = [f"eq{i}" for i in range(10)]
        build_index(vectors, labels)
        assert index_size() == 10

    def test_size_after_clear(self) -> None:
        from app.faiss_index import build_index
        vectors = np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)
        build_index(vectors, [f"v{i}" for i in range(5)])
        clear_index()
        assert index_size() == 0


class TestClearIndex:
    def test_clear_on_empty_no_error(self) -> None:
        clear_index()

    def test_clear_resets_size(self) -> None:
        from app.faiss_index import build_index
        vectors = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
        build_index(vectors, ["a", "b", "c"])
        clear_index()
        assert index_size() == 0

    def test_clear_then_rebuild(self) -> None:
        from app.faiss_index import build_index
        v1 = np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)
        build_index(v1, [f"old{i}" for i in range(5)])
        clear_index()
        v2 = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
        build_index(v2, ["new0", "new1", "new2"])
        assert index_size() == 3

    def test_search_after_clear_returns_empty(self) -> None:
        from app.faiss_index import search_similar
        clear_index()
        result = search_similar(np.zeros(4, dtype=np.float32))
        assert result == []
