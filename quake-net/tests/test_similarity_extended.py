"""Extended tests for quake-net app/similarity.py."""

from __future__ import annotations

import numpy as np
import pytest

from app.similarity import (
    SIGNATURE_COLUMNS,
    SeismicIndex,
    build_signature_matrix,
    faiss_available,
)


class TestFaissAvailable:
    def test_returns_bool(self) -> None:
        assert isinstance(faiss_available(), bool)


class TestSignatureColumns:
    def test_not_empty(self) -> None:
        assert len(SIGNATURE_COLUMNS) > 0

    def test_all_strings(self) -> None:
        for col in SIGNATURE_COLUMNS:
            assert isinstance(col, str)


class TestBuildSignatureMatrix:
    def _record(self, **kwargs) -> dict:
        base = {col: 1.0 for col in SIGNATURE_COLUMNS}
        base.update(kwargs)
        return base

    def test_basic_output_shape(self) -> None:
        records = [self._record(), self._record()]
        matrix = build_signature_matrix(records)
        assert matrix.shape == (2, len(SIGNATURE_COLUMNS))

    def test_dtype_float32(self) -> None:
        matrix = build_signature_matrix([self._record()])
        assert matrix.dtype == np.float32

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError):
            build_signature_matrix([])

    def test_missing_column_defaults_to_zero(self) -> None:
        record = {}
        matrix = build_signature_matrix([record])
        assert matrix.shape == (1, len(SIGNATURE_COLUMNS))

    def test_log_transform_applied(self) -> None:
        record = {col: 0.0 for col in SIGNATURE_COLUMNS}
        matrix = build_signature_matrix([record])
        assert float(matrix[0, 0]) == pytest.approx(np.log1p(0.0))


class TestSeismicIndex:
    def _records(self, n: int = 5) -> list[dict]:
        return [{col: float(i) for col in SIGNATURE_COLUMNS} for i in range(n)]

    def test_initial_size_zero(self) -> None:
        idx = SeismicIndex()
        assert idx.size == 0

    def test_initial_dimension_none(self) -> None:
        idx = SeismicIndex()
        assert idx.dimension is None

    def test_build_sets_size(self) -> None:
        idx = SeismicIndex()
        idx.build(self._records(5))
        assert idx.size == 5

    def test_build_sets_dimension(self) -> None:
        idx = SeismicIndex()
        idx.build(self._records(5))
        assert idx.dimension == len(SIGNATURE_COLUMNS)

    def test_search_returns_list(self) -> None:
        idx = SeismicIndex()
        idx.build(self._records(5))
        query = {col: 1.0 for col in SIGNATURE_COLUMNS}
        results = idx.search(query, k=2)
        assert isinstance(results, list)

    def test_search_k_limits_results(self) -> None:
        idx = SeismicIndex()
        idx.build(self._records(10))
        query = {col: 1.0 for col in SIGNATURE_COLUMNS}
        results = idx.search(query, k=3)
        assert len(results) <= 3

    def test_search_on_empty_index_returns_empty(self) -> None:
        idx = SeismicIndex()
        query = {col: 1.0 for col in SIGNATURE_COLUMNS}
        results = idx.search(query, k=3)
        assert results == []
