"""Tests for RAG ingest and retriever."""

from __future__ import annotations

import numpy as np


class TestBuildPatternVector:
    def test_returns_float32_array(self):
        from rag.ingest import build_pattern_vector

        vec = build_pattern_vector({"consumption_kwh": 4.5, "temperature_c": 22.0})
        assert vec.dtype == np.float32

    def test_vector_length_is_6(self):
        from rag.ingest import build_pattern_vector

        vec = build_pattern_vector({})
        assert len(vec) == 6

    def test_consumption_reflected_in_vector(self):
        from rag.ingest import build_pattern_vector

        vec = build_pattern_vector({"consumption_kwh": 99.0})
        assert vec[0] == 99.0


class TestRetriever:
    def test_find_similar_no_index_returns_empty(self):
        from rag import retriever

        retriever._index = None
        retriever._meta = []
        result = retriever.find_similar_patterns({"consumption_kwh": 4.5})
        assert result == []

    def test_baseline_consumption_fallback(self):
        from rag import retriever

        retriever._index = None
        retriever._meta = []
        baseline = retriever.get_baseline_consumption(12, 2, "residential")
        assert baseline == 3.5
