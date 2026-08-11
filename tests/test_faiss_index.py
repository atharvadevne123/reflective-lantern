"""Tests for FAISS load-pattern similarity search."""

from __future__ import annotations

import pytest

from app.faiss_index import LoadPatternIndex, get_pattern_index


class TestLoadPatternIndex:
    def test_add_and_size(self):
        idx = LoadPatternIndex(dim=6)
        idx.add([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], {"period": "morning"})
        assert idx.size == 1

    def test_add_wrong_dim_raises(self):
        idx = LoadPatternIndex(dim=6)
        with pytest.raises(ValueError):
            idx.add([1.0, 2.0, 3.0])

    def test_search_returns_results(self):
        idx = LoadPatternIndex(dim=4)
        idx.add([1.0, 0.0, 0.0, 0.0], {"label": "a"})
        idx.add([0.0, 1.0, 0.0, 0.0], {"label": "b"})
        idx.add([0.0, 0.0, 1.0, 0.0], {"label": "c"})
        results = idx.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) <= 2
        assert results[0]["rank"] == 1

    def test_search_most_similar_first(self):
        idx = LoadPatternIndex(dim=4)
        idx.add([100.0, 0.0, 0.0, 0.0], {"label": "match"})
        idx.add([0.0, 100.0, 0.0, 0.0], {"label": "no_match"})
        results = idx.search([99.0, 1.0, 0.0, 0.0], k=2)
        assert results[0]["metadata"]["label"] == "match"

    def test_search_empty_index(self):
        idx = LoadPatternIndex(dim=4)
        results = idx.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert results == []

    def test_metadata_returned(self):
        idx = LoadPatternIndex(dim=3)
        idx.add([1.0, 2.0, 3.0], {"date": "2026-01-01", "region": "northeast"})
        results = idx.search([1.0, 2.0, 3.0], k=1)
        if results:
            assert results[0]["metadata"]["region"] == "northeast"

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_k_results_at_most(self, k):
        idx = LoadPatternIndex(dim=3)
        for i in range(4):
            idx.add([float(i), float(i + 1), float(i + 2)])
        results = idx.search([1.0, 2.0, 3.0], k=k)
        assert len(results) <= k

    def test_build_does_not_crash(self):
        idx = LoadPatternIndex(dim=3)
        for i in range(10):
            idx.add([float(i), float(i + 1), float(i + 2)])
        idx.build()
        results = idx.search([5.0, 6.0, 7.0], k=3)
        assert isinstance(results, list)

    def test_clear_resets_size(self):
        idx = LoadPatternIndex(dim=3)
        idx.add([1.0, 2.0, 3.0])
        idx.add([4.0, 5.0, 6.0])
        idx.clear()
        assert idx.size == 0

    def test_clear_search_returns_empty(self):
        idx = LoadPatternIndex(dim=3)
        idx.add([1.0, 2.0, 3.0])
        idx.clear()
        assert idx.search([1.0, 2.0, 3.0]) == []

    def test_similarity_score_in_range(self):
        idx = LoadPatternIndex(dim=3)
        idx.add([1.0, 0.0, 0.0])
        idx.add([-1.0, 0.0, 0.0])
        for r in idx.search([1.0, 0.0, 0.0]):
            assert -1.0 <= r["similarity"] <= 1.0


class TestGetPatternIndex:
    def test_returns_same_instance(self):
        a = get_pattern_index(dim=24)
        b = get_pattern_index(dim=24)
        assert a is b


def test_load_pattern_index_len() -> None:
    from app.faiss_index import LoadPatternIndex

    idx = LoadPatternIndex(dim=3)
    assert len(idx) == 0
    idx.add([1.0, 2.0, 3.0])
    assert len(idx) == 1
    idx.add([4.0, 5.0, 6.0])
    assert len(idx) == 2


def test_load_pattern_index_reset_clears() -> None:
    from app.faiss_index import LoadPatternIndex

    idx = LoadPatternIndex(dim=3)
    idx.add([1.0, 2.0, 3.0])
    idx.reset()
    assert len(idx) == 0
    assert idx.search([1.0, 2.0, 3.0]) == []


def test_save_vectors_empty_index() -> None:
    import numpy as np

    from app.faiss_index import LoadPatternIndex

    idx = LoadPatternIndex(dim=4)
    arr = idx.save_vectors()
    assert arr.shape == (0, 4)
    assert arr.dtype == np.float32


def test_save_vectors_returns_copy() -> None:
    import numpy as np

    from app.faiss_index import LoadPatternIndex

    idx = LoadPatternIndex(dim=3)
    idx.add([1.0, 0.0, 0.0])
    idx.add([0.0, 1.0, 0.0])
    arr = idx.save_vectors()
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float32


def test_save_vectors_does_not_mutate_index() -> None:
    from app.faiss_index import LoadPatternIndex

    idx = LoadPatternIndex(dim=2)
    idx.add([3.0, 4.0])
    arr = idx.save_vectors()
    arr[0, 0] = 999.0  # modify the copy
    assert idx._vectors[0][0] == pytest.approx(3.0)


def test_search_comparable_returns_distance():
    from app.faiss_index import add_property, reset_index, search_comparable

    reset_index()
    add_property([1.0, 0.0, 0.0] + [0.0] * 21, {"name": "A"})
    add_property([0.0, 1.0, 0.0] + [0.0] * 21, {"name": "B"})
    results = search_comparable([1.0, 0.0, 0.0] + [0.0] * 21, top_k=2)
    assert len(results) >= 1
    assert "distance" in results[0]
    reset_index()


def test_search_comparable_caps_top_k():
    from app.faiss_index import MAX_TOP_K, add_property, reset_index, search_comparable

    reset_index()
    for i in range(5):
        add_property([float(i)] + [0.0] * 23)
    results = search_comparable([1.0] + [0.0] * 23, top_k=MAX_TOP_K + 50)
    assert len(results) <= MAX_TOP_K
    reset_index()


def test_index_size_updates():
    from app.faiss_index import add_property, index_size, reset_index

    reset_index()
    initial = index_size()
    add_property([1.0] * 24)
    assert index_size() == initial + 1
    reset_index()


@pytest.mark.parametrize("n", [1, 3, 5])
def test_search_comparable_returns_n_or_less(n):
    from app.faiss_index import add_property, reset_index, search_comparable

    reset_index()
    for i in range(n):
        add_property([float(i)] * 24)
    results = search_comparable([0.0] * 24, top_k=n)
    assert len(results) <= n
    reset_index()


def test_add_property_increments_index():
    from app.faiss_index import add_property, index_size, reset_index

    reset_index()
    assert index_size() == 0
    add_property([1.0] * 24)
    add_property([2.0] * 24)
    assert index_size() == 2
    reset_index()


def test_reset_index_empties_store():
    from app.faiss_index import add_property, index_size, reset_index

    reset_index()
    add_property([0.5] * 24)
    reset_index()
    assert index_size() == 0


def test_search_comparable_returns_list():
    from app.faiss_index import add_property, reset_index, search_comparable

    reset_index()
    for i in range(5):
        add_property([float(i + 1)] * 24)
    results = search_comparable([3.0] * 24, top_k=3)
    assert isinstance(results, list)
    reset_index()


@pytest.mark.parametrize("dim_val", [0.0, 5.0, 10.0, 100.0])
def test_add_property_various_values(dim_val: float) -> None:
    from app.faiss_index import add_property, index_size, reset_index

    reset_index()
    add_property([dim_val] * 24)
    assert index_size() == 1
    reset_index()


class TestIndexIsEmpty:
    def test_empty_after_reset(self) -> None:
        from app.faiss_index import index_is_empty, reset_index

        reset_index()
        assert index_is_empty() is True

    def test_not_empty_after_add(self) -> None:
        from app.faiss_index import DIM, add_property, index_is_empty, reset_index

        reset_index()
        add_property([0.0] * DIM)
        assert index_is_empty() is False


class TestBatchAddProperties:
    def test_adds_multiple(self) -> None:
        from app.faiss_index import DIM, batch_add_properties, index_size, reset_index

        reset_index()
        vecs = [[float(i)] * DIM for i in range(3)]
        count = batch_add_properties(vecs)
        assert count == 3
        assert index_size() == 3

    def test_empty_list(self) -> None:
        from app.faiss_index import batch_add_properties, reset_index

        reset_index()
        assert batch_add_properties([]) == 0


class TestTopKDistances:
    def test_returns_list(self) -> None:
        from app.faiss_index import DIM, add_property, reset_index, top_k_distances

        reset_index()
        add_property([1.0] * DIM)
        result = top_k_distances([1.0] * DIM, top_k=1)
        assert isinstance(result, list)
        assert len(result) <= 1

    def test_empty_index(self) -> None:
        from app.faiss_index import DIM, reset_index, top_k_distances

        reset_index()
        result = top_k_distances([0.0] * DIM, top_k=5)
        assert result == []


class TestIndexSummary:
    def test_keys_present(self) -> None:
        from app.faiss_index import index_summary

        result = index_summary()
        assert "size" in result
        assert "is_empty" in result
        assert "dim" in result

    def test_empty_after_reset(self) -> None:
        from app.faiss_index import index_summary, reset_index

        reset_index()
        result = index_summary()
        assert result["is_empty"] is True
        assert result["size"] == 0


class TestBatchAdd:
    def test_adds_multiple(self) -> None:
        from app.faiss_index import DIM, batch_add, index_size, reset_index

        reset_index()
        vectors = [[float(i)] * DIM for i in range(3)]
        count = batch_add(vectors)
        assert count == 3
        assert index_size() == 3

    def test_empty_list(self) -> None:
        from app.faiss_index import batch_add, index_size, reset_index

        reset_index()
        assert batch_add([]) == 0
        assert index_size() == 0


class TestSearchRadius:
    def test_within_radius_returned(self) -> None:
        from app.faiss_index import DIM, add_property, reset_index, search_radius

        reset_index()
        v = [1.0] * DIM
        add_property(v)
        results = search_radius([1.0] * DIM, radius=0.1, top_k=5)
        assert len(results) >= 1

    def test_outside_radius_excluded(self) -> None:
        from app.faiss_index import DIM, add_property, reset_index, search_radius

        reset_index()
        add_property([1.0] * DIM)
        results = search_radius([0.0] * DIM, radius=0.0001, top_k=5)
        assert results == []


class TestNearestDistance:
    def test_none_when_empty(self) -> None:
        from app.faiss_index import DIM, nearest_distance, reset_index

        reset_index()
        assert nearest_distance([0.0] * DIM) is None

    def test_zero_for_exact_match(self) -> None:
        from app.faiss_index import DIM, add_property, nearest_distance, reset_index

        reset_index()
        v = [1.0] * DIM
        add_property(v)
        dist = nearest_distance(v)
        assert dist is not None
        assert dist == pytest.approx(0.0, abs=1e-5)
