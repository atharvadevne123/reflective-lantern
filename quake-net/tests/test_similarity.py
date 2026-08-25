"""Tests for the seismic signature similarity index."""

from __future__ import annotations

import numpy as np
import pytest

from app.similarity import (
    SIGNATURE_COLUMNS,
    SeismicIndex,
    build_signature_matrix,
    faiss_available,
    get_index,
)


def _record(
    depth: float = 10.0,
    p: float = 3.0,
    s: float = 6.0,
    dist: float = 80.0,
    stations: int = 12,
    **extra,
) -> dict:
    return {
        "depth_km": depth,
        "p_wave_amplitude": p,
        "s_wave_amplitude": s,
        "epicentral_distance_km": dist,
        "station_count": stations,
        **extra,
    }


class TestBuildSignatureMatrix:
    def test_shape_matches_signature_columns(self) -> None:
        matrix = build_signature_matrix([_record(), _record(depth=50.0)])
        assert matrix.shape == (2, len(SIGNATURE_COLUMNS))

    def test_dtype_is_float32(self) -> None:
        assert build_signature_matrix([_record()]).dtype == np.float32

    def test_values_are_finite(self) -> None:
        matrix = build_signature_matrix([_record(p=1e9, dist=1e6)])
        assert np.isfinite(matrix).all()

    def test_missing_column_defaults_to_zero(self) -> None:
        matrix = build_signature_matrix([{"depth_km": 10.0}])
        assert matrix.shape == (1, len(SIGNATURE_COLUMNS))

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError, match="zero records"):
            build_signature_matrix([])

    def test_negative_values_clipped_not_nan(self) -> None:
        matrix = build_signature_matrix([_record(depth=-5.0)])
        assert np.isfinite(matrix).all()


class TestSeismicIndex:
    def test_size_reflects_record_count(self) -> None:
        index = SeismicIndex().build([_record(depth=d) for d in (1.0, 20.0, 90.0)])
        assert index.size == 3

    def test_dimension_matches_signature_width(self) -> None:
        index = SeismicIndex().build([_record()])
        assert index.dimension == len(SIGNATURE_COLUMNS)

    def test_empty_index_has_no_dimension(self) -> None:
        assert SeismicIndex().dimension is None
        assert SeismicIndex().size == 0

    def test_search_before_build_raises(self) -> None:
        with pytest.raises(ValueError, match="before build"):
            SeismicIndex().search(_record())

    def test_search_returns_requested_count(self) -> None:
        index = SeismicIndex().build([_record(depth=float(d)) for d in range(1, 11)])
        assert len(index.search(_record(depth=5.0), k=3)) == 3

    def test_search_k_capped_at_index_size(self) -> None:
        index = SeismicIndex().build([_record(), _record(depth=40.0)])
        assert len(index.search(_record(), k=50)) == 2

    def test_nearest_match_is_the_identical_record(self) -> None:
        target = _record(depth=11.0, p=3.1, s=6.1, id=99)
        index = SeismicIndex().build([_record(depth=500.0, p=99.0, id=1), target])
        assert index.search(target, k=1)[0]["id"] == 99

    def test_results_carry_distance_and_similarity(self) -> None:
        index = SeismicIndex().build([_record(), _record(depth=40.0)])
        match = index.search(_record(), k=1)[0]
        assert "distance" in match
        assert 0.0 < match["similarity"] <= 1.0

    def test_results_sorted_by_ascending_distance(self) -> None:
        index = SeismicIndex().build([_record(depth=float(d)) for d in (1, 30, 200, 600)])
        matches = index.search(_record(depth=1.0), k=4)
        distances = [m["distance"] for m in matches]
        assert distances == sorted(distances)

    def test_payload_fields_preserved(self) -> None:
        index = SeismicIndex().build([_record(id=7, fault_type="reverse")])
        match = index.search(_record(), k=1)[0]
        assert match["id"] == 7
        assert match["fault_type"] == "reverse"

    def test_build_does_not_persist_by_default(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "should_not_exist.faiss"
        monkeypatch.setattr("app.similarity.INDEX_PATH", target)
        SeismicIndex().build([_record()])
        assert not target.exists()

    def test_rebuild_replaces_previous_records(self) -> None:
        index = SeismicIndex().build([_record(id=1), _record(id=2)])
        index.build([_record(id=3)])
        assert index.size == 1
        assert index.search(_record(), k=1)[0]["id"] == 3


class TestModuleIndex:
    def test_get_index_is_singleton(self) -> None:
        assert get_index() is get_index()

    def test_faiss_available_returns_bool(self) -> None:
        assert isinstance(faiss_available(), bool)
