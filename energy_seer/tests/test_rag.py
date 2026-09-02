"""Tests for RAG ingest and retriever."""

from __future__ import annotations

import numpy as np
import pytest


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


class TestBuildPatternVectorEdgeCases:
    @pytest.mark.parametrize("consumption", [0.0, 1.5, 100.0])
    def test_consumption_kwh_stored_at_index_zero(self, consumption: float) -> None:
        from rag.ingest import build_pattern_vector

        vec = build_pattern_vector({"consumption_kwh": consumption})
        assert vec[0] == consumption

    def test_missing_keys_default_to_zero(self) -> None:
        from rag.ingest import build_pattern_vector

        vec = build_pattern_vector({})
        assert all(v == 0.0 for v in vec)

    def test_unknown_keys_ignored(self) -> None:
        from rag.ingest import build_pattern_vector

        vec = build_pattern_vector({"unknown_key": 99.9})
        assert len(vec) == 6


class TestBaselineConsumption:
    @pytest.mark.parametrize(
        "hour,month,segment,expected",
        [
            (12, 2, "residential", 3.5),
            (0, 6, "commercial", 3.5),
            (3, 11, "industrial", 3.5),
        ],
    )
    def test_fallback_returns_constant(
        self, hour: int, month: int, segment: str, expected: float
    ) -> None:
        from rag import retriever

        retriever._index = None
        retriever._meta = []
        result = retriever.get_baseline_consumption(hour, month, segment)
        assert result == expected

    def test_returns_float(self) -> None:
        from rag import retriever

        retriever._index = None
        retriever._meta = []
        assert isinstance(retriever.get_baseline_consumption(0, 1, "residential"), float)


class TestBuildVectorTypes:
    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 1e6])
    def test_extreme_consumption_values_stored(self, value: float) -> None:
        from rag.ingest import build_pattern_vector

        vec = build_pattern_vector({"consumption_kwh": value})
        assert vec[0] == value
