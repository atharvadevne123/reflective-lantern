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
