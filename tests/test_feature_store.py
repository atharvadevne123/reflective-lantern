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

    def test_empty_store_list_names_empty(self):
        store = FeatureStore()
        assert store.list_names() == []

    def test_empty_store_list_versions_empty(self):
        store = FeatureStore()
        assert store.list_versions("energy") == []

    def test_feature_values_preserved_exactly(self):
        features = {"a": 1.5, "b": "text", "c": [1, 2, 3], "d": True}
        store = FeatureStore()
        store.publish(_fs(features=features))
        result = store.get_latest("energy")
        assert result.get("a") == 1.5
        assert result.get("b") == "text"
        assert result.get("d") is True

    def test_delete_specific_version_leaves_others(self):
        store = FeatureStore()
        for v in ["1.0.0", "2.0.0", "3.0.0"]:
            store.publish(_fs(version=v))
        store.delete("energy", "2.0.0")
        remaining = store.list_versions("energy")
        assert "2.0.0" not in remaining
        assert len(remaining) == 2

    def test_publish_multiple_feature_sets_same_version_different_name(self):
        store = FeatureStore()
        store.publish(_fs(name="setA", version="1.0.0"))
        store.publish(_fs(name="setB", version="1.0.0"))
        assert store.get_latest("setA") is not None
        assert store.get_latest("setB") is not None


@pytest.mark.parametrize("n_versions", [1, 3, 5])
def test_list_versions_count_matches_published(n_versions: int) -> None:
    """list_versions returns one entry per published version."""
    from app.feature_store import FeatureSet, FeatureStore

    store = FeatureStore()
    for i in range(n_versions):
        store.publish(FeatureSet(name="ds", version=f"{i}.0.0", features={"f": [1.0]}))
    assert len(store.list_versions("ds")) == n_versions


@pytest.mark.parametrize("n_names", [1, 3, 5])
def test_list_datasets_includes_all_published_names(n_names: int) -> None:
    """Every distinct name published appears in list_datasets."""
    from app.feature_store import FeatureSet, FeatureStore

    store = FeatureStore()
    names = [f"dataset_{i}" for i in range(n_names)]
    for name in names:
        store.publish(FeatureSet(name=name, version="1.0.0", features={"x": [0.0]}))
    listed = store.list_datasets()
    assert all(name in listed for name in names)


class TestFeatureSetEdgeCases:
    def test_feature_names_preserved(self) -> None:
        from app.feature_store import FeatureSet

        fs = FeatureSet(name="my_ds", version="1.0.0", features={"a": [1.0], "b": [2.0]})
        assert set(fs.features.keys()) == {"a", "b"}

    @pytest.mark.parametrize("version", ["1.0.0", "2.3.4", "0.0.1"])
    def test_version_string_preserved(self, version: str) -> None:
        from app.feature_store import FeatureSet

        fs = FeatureSet(name="v_test", version=version, features={})
        assert fs.version == version
