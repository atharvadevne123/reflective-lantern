"""Tests for app.feature_store module."""

from __future__ import annotations

import pytest

from app.feature_store import FeatureSet, FeatureStore


def _fs(name="energy", version="1.0.0", features=None) -> FeatureSet:
    return FeatureSet(name=name, version=version, features=features or {"a": 1})


class TestFeatureSet:
    def test_get_returns_value(self):
        fs = _fs(features={"x": 42})
        assert fs.get("x") == 42

    def test_get_returns_default(self):
        fs = _fs(features={})
        assert fs.get("missing", "default") == "default"

    def test_keys_returns_all_names(self):
        fs = _fs(features={"a": 1, "b": 2})
        assert set(fs.keys()) == {"a", "b"}


class TestFeatureStore:
    def test_publish_and_get_latest(self):
        store = FeatureStore()
        store.publish(_fs())
        result = store.get_latest("energy")
        assert result is not None
        assert result.version == "1.0.0"

    def test_latest_returns_most_recent(self):
        store = FeatureStore()
        store.publish(_fs(version="1.0.0"))
        store.publish(_fs(version="2.0.0"))
        assert store.get_latest("energy").version == "2.0.0"

    def test_duplicate_version_raises(self):
        store = FeatureStore()
        store.publish(_fs(version="1.0.0"))
        with pytest.raises(ValueError, match="already exists"):
            store.publish(_fs(version="1.0.0"))

    def test_get_version_specific(self):
        store = FeatureStore()
        store.publish(_fs(version="1.0.0"))
        store.publish(_fs(version="2.0.0"))
        result = store.get_version("energy", "1.0.0")
        assert result.version == "1.0.0"

    def test_get_version_not_found_returns_none(self):
        store = FeatureStore()
        assert store.get_version("energy", "9.9.9") is None

    def test_get_latest_unknown_returns_none(self):
        store = FeatureStore()
        assert store.get_latest("unknown") is None

    def test_list_versions(self):
        store = FeatureStore()
        store.publish(_fs(version="1.0.0"))
        store.publish(_fs(version="1.1.0"))
        assert store.list_versions("energy") == ["1.0.0", "1.1.0"]

    def test_list_names(self):
        store = FeatureStore()
        store.publish(_fs(name="a"))
        store.publish(_fs(name="b"))
        assert set(store.list_names()) == {"a", "b"}

    def test_delete_all_versions(self):
        store = FeatureStore()
        store.publish(_fs())
        assert store.delete("energy") is True
        assert store.get_latest("energy") is None

    def test_delete_specific_version(self):
        store = FeatureStore()
        store.publish(_fs(version="1.0.0"))
        store.publish(_fs(version="2.0.0"))
        store.delete("energy", "1.0.0")
        assert store.list_versions("energy") == ["2.0.0"]

    def test_delete_nonexistent_returns_false(self):
        store = FeatureStore()
        assert store.delete("ghost") is False

    @pytest.mark.parametrize("version", ["1.0.0", "2.0.0", "3.0.0"])
    def test_multiple_versions_retrievable(self, version):
        store = FeatureStore()
        for v in ["1.0.0", "2.0.0", "3.0.0"]:
            store.publish(_fs(version=v, features={"v": v}))
        result = store.get_version("energy", version)
        assert result.features["v"] == version


class TestFeatureStoreExtensions:
    def test_version_count(self):
        store = FeatureStore()
        store.publish(_fs(version="1.0.0"))
        store.publish(_fs(version="2.0.0"))
        assert store.version_count("energy") == 2

    def test_version_count_unknown(self):
        assert FeatureStore().version_count("ghost") == 0

    def test_len_counts_names(self):
        store = FeatureStore()
        assert len(store) == 0
        store.publish(_fs(name="a"))
        store.publish(_fs(name="b"))
        assert len(store) == 2

    def test_total_versions(self):
        store = FeatureStore()
        store.publish(_fs(name="a", version="1.0.0"))
        store.publish(_fs(name="a", version="2.0.0"))
        store.publish(_fs(name="b", version="1.0.0"))
        assert store.total_versions() == 3

    def test_total_versions_empty(self):
        assert FeatureStore().total_versions() == 0
