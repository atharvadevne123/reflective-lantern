"""Tests for app.data_versioning module."""

from __future__ import annotations

import pytest

from app.data_versioning import DataLineage, DataSnapshot


def _snap(name="energy", version="1.0.0", **kwargs) -> DataSnapshot:
    return DataSnapshot(name=name, version=version, source="s3://data/raw", **kwargs)


class TestDataSnapshot:
    def test_compute_checksum_returns_hex(self):
        checksum = DataSnapshot.compute_checksum({"a": 1})
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_same_data_same_checksum(self):
        c1 = DataSnapshot.compute_checksum([1, 2, 3])
        c2 = DataSnapshot.compute_checksum([1, 2, 3])
        assert c1 == c2

    def test_different_data_different_checksum(self):
        c1 = DataSnapshot.compute_checksum({"a": 1})
        c2 = DataSnapshot.compute_checksum({"a": 2})
        assert c1 != c2

    def test_defaults_populated(self):
        snap = _snap()
        assert snap.row_count == 0
        assert snap.schema == {}
        assert snap.tags == {}
        assert snap.parent_versions == []


class TestDataLineage:
    def test_record_and_retrieve(self):
        dl = DataLineage()
        dl.record(_snap())
        result = dl.get("energy")
        assert result.version == "1.0.0"

    def test_get_latest_returns_last(self):
        dl = DataLineage()
        dl.record(_snap(version="1.0.0"))
        dl.record(_snap(version="2.0.0"))
        assert dl.get("energy").version == "2.0.0"

    def test_get_specific_version(self):
        dl = DataLineage()
        dl.record(_snap(version="1.0.0"))
        dl.record(_snap(version="2.0.0"))
        assert dl.get("energy", "1.0.0").version == "1.0.0"

    def test_duplicate_version_raises(self):
        dl = DataLineage()
        dl.record(_snap())
        with pytest.raises(ValueError, match="already recorded"):
            dl.record(_snap())

    def test_unknown_name_returns_none(self):
        dl = DataLineage()
        assert dl.get("ghost") is None

    def test_unknown_version_returns_none(self):
        dl = DataLineage()
        dl.record(_snap())
        assert dl.get("energy", "9.9.9") is None

    def test_lineage_traces_parents(self):
        dl = DataLineage()
        dl.record(_snap(version="1.0.0"))
        dl.record(_snap(version="2.0.0", parent_versions=["1.0.0"]))
        ancestors = dl.lineage("energy", "2.0.0")
        assert any(a.version == "1.0.0" for a in ancestors)

    def test_lineage_empty_for_root(self):
        dl = DataLineage()
        dl.record(_snap(version="1.0.0"))
        assert dl.lineage("energy", "1.0.0") == []

    def test_list_versions(self):
        dl = DataLineage()
        dl.record(_snap(version="1.0.0"))
        dl.record(_snap(version="1.1.0"))
        assert dl.list_versions("energy") == ["1.0.0", "1.1.0"]

    def test_list_datasets(self):
        dl = DataLineage()
        dl.record(_snap(name="a"))
        dl.record(_snap(name="b"))
        assert set(dl.list_datasets()) == {"a", "b"}

    @pytest.mark.parametrize("rows", [0, 100, 1_000_000])
    def test_row_count_stored(self, rows):
        dl = DataLineage()
        dl.record(_snap(row_count=rows))
        assert dl.get("energy").row_count == rows

    def test_list_datasets_empty_when_none_recorded(self) -> None:
        dl = DataLineage()
        assert dl.list_datasets() == []

    def test_list_versions_empty_for_unknown(self) -> None:
        dl = DataLineage()
        assert dl.list_versions("nonexistent") == []

    def test_schema_stored_and_retrieved(self) -> None:
        schema = {"col_a": "float", "col_b": "int"}
        dl = DataLineage()
        dl.record(_snap(schema=schema))
        assert dl.get("energy").schema == schema
