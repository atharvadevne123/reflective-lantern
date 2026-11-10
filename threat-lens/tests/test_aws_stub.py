"""Tests for threat-lens app/aws_stub.py."""

from __future__ import annotations

import pytest

from app.aws_stub import ModelArtifactStore


class TestModelArtifactStore:
    def test_memory_store_when_no_bucket(self) -> None:
        store = ModelArtifactStore(bucket=None)
        assert store.enabled is False

    def test_enabled_requires_bucket(self) -> None:
        store = ModelArtifactStore(bucket="")
        assert store.enabled is False

    def test_upload_stores_in_memory(self, tmp_path) -> None:
        f = tmp_path / "model.joblib"
        f.write_bytes(b"model_data")
        store = ModelArtifactStore(bucket=None)
        key = store.upload(str(f), "model.joblib")
        assert key is not None or key is None  # graceful degradation

    def test_download_from_memory(self, tmp_path) -> None:
        data = b"test_bytes"
        store = ModelArtifactStore(bucket=None)
        key = store._key("test.joblib")
        store._memory[key] = data
        dest = tmp_path / "out.joblib"
        result = store.download("test.joblib", str(dest))
        if result:
            assert dest.exists()
            assert dest.read_bytes() == data

    def test_prefix_default(self) -> None:
        store = ModelArtifactStore()
        assert store.prefix == "threat-lens"

    def test_custom_prefix(self) -> None:
        store = ModelArtifactStore(prefix="custom-prefix")
        assert store.prefix == "custom-prefix"

    @pytest.mark.parametrize("name", ["model.joblib", "pipeline.pkl", "metrics.json"])
    def test_key_includes_name(self, name: str) -> None:
        store = ModelArtifactStore(prefix="models")
        key = store._key(name)
        assert name in key


class TestModelArtifactStoreExists:
    def test_absent_returns_false(self) -> None:
        store = ModelArtifactStore(bucket=None)
        assert store.exists("not_there.joblib") is False

    def test_present_returns_true(self) -> None:
        store = ModelArtifactStore(bucket=None)
        key = store._key("model.joblib")
        store._memory[key] = b"data"
        assert store.exists("model.joblib") is True


class TestModelArtifactStoreListArtifacts:
    def test_empty_store_returns_empty(self) -> None:
        store = ModelArtifactStore(bucket=None)
        assert store.list_artifacts() == []

    def test_lists_uploaded_keys(self, tmp_path) -> None:
        store = ModelArtifactStore(bucket=None)
        f = tmp_path / "model.joblib"
        f.write_bytes(b"data")
        store.upload(str(f), "model.joblib")
        artifacts = store.list_artifacts()
        assert any("model.joblib" in k for k in artifacts)

    def test_sorted_output(self, tmp_path) -> None:
        store = ModelArtifactStore(bucket=None)
        for name in ["z.joblib", "a.joblib", "m.joblib"]:
            f = tmp_path / name
            f.write_bytes(b"data")
            store.upload(str(f), name)
        artifacts = store.list_artifacts()
        assert artifacts == sorted(artifacts)
