"""Tests for Fleet-Mind in-process feature store."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.feature_store as fs_module
from app.feature_store import evict_expired, get_features, store_features


@pytest.fixture(autouse=True)
def clear_store():
    fs_module._STORE.clear()
    yield
    fs_module._STORE.clear()


def test_store_and_retrieve_features():
    features = np.array([1.0, 2.0, 3.0])
    store_features("VH-001", features)
    result = get_features("VH-001")
    assert result is not None
    np.testing.assert_array_equal(result, features)


def test_missing_vehicle_returns_none():
    result = get_features("MISSING-VH")
    assert result is None


def test_expired_entry_returns_none():
    features = np.array([1.0])
    store_features("VH-EXP", features, ttl_seconds=0.01)
    time.sleep(0.05)
    result = get_features("VH-EXP")
    assert result is None


def test_non_expired_entry_returns_features():
    features = np.array([5.0, 6.0])
    store_features("VH-TTL", features, ttl_seconds=100.0)
    result = get_features("VH-TTL")
    assert result is not None


def test_overwrite_existing_entry():
    store_features("VH-X", np.array([1.0]))
    store_features("VH-X", np.array([9.0]))
    result = get_features("VH-X")
    np.testing.assert_array_equal(result, np.array([9.0]))


def test_evict_expired_removes_stale_entries():
    store_features("VH-A", np.array([1.0]), ttl_seconds=0.01)
    store_features("VH-B", np.array([2.0]), ttl_seconds=100.0)
    time.sleep(0.05)
    count = evict_expired()
    assert count >= 1
    assert get_features("VH-B") is not None


def test_evict_expired_returns_int():
    count = evict_expired()
    assert isinstance(count, int)


def test_store_multiple_vehicles():
    for i in range(5):
        store_features(f"VH-{i:03d}", np.array([float(i)]))
    for i in range(5):
        result = get_features(f"VH-{i:03d}")
        assert result is not None
        assert result[0] == float(i)
