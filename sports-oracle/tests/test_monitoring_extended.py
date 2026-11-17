"""Extended tests for Sports-Oracle drift detection utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.monitoring import compute_drift, compute_psi


def test_compute_drift_returns_dict():
    ref = list(range(20))
    cur = list(range(20))
    result = compute_drift(ref, cur)
    assert isinstance(result, dict)


def test_compute_drift_has_required_keys():
    ref = list(range(20))
    cur = list(range(20))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result


def test_compute_drift_insufficient_samples_warning():
    result = compute_drift([1.0, 2.0], [1.0, 2.0])
    assert "warning" in result
    assert result["drift_detected"] is False


def test_compute_drift_no_drift_same_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 50).tolist()
    cur = rng.normal(0, 1, 50).tolist()
    result = compute_drift(ref, cur)
    assert isinstance(result["ks_statistic"], float)
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_compute_drift_detects_clear_drift():
    ref = list(range(50))
    cur = [x + 100 for x in range(50)]
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True


def test_compute_drift_p_value_in_range():
    ref = list(range(20))
    cur = list(range(20))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["p_value"] <= 1.0


def test_compute_psi_returns_float():
    ref = np.arange(100, dtype=float)
    cur = np.arange(100, dtype=float)
    result = compute_psi(ref, cur)
    assert isinstance(result, float)


def test_compute_psi_similar_distributions_low():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(0, 1, 500)
    result = compute_psi(ref, cur)
    assert result < 1.0


def test_compute_psi_different_distributions_higher():
    ref = np.zeros(100)
    cur = np.ones(100) * 10
    result = compute_psi(ref, cur)
    assert isinstance(result, float)


@pytest.mark.parametrize("size", [20, 50, 100])
def test_compute_drift_various_sizes(size):
    ref = list(range(size))
    cur = list(range(size))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert isinstance(result["drift_detected"], bool)
