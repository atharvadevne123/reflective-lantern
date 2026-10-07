"""Drift detection and monitoring tests for Sports-Oracle."""

from __future__ import annotations

import numpy as np
import pytest


def test_compute_drift_no_drift():
    from app.monitoring import compute_drift

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 200).tolist()
    cur = rng.normal(0, 1, 100).tolist()
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert result["p_value"] >= 0.0


def test_compute_drift_detects_shift():
    from app.monitoring import compute_drift

    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 300).tolist()
    cur = rng.normal(5, 1, 300).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_samples():
    from app.monitoring import compute_drift

    result = compute_drift([0.1, 0.2], [0.3, 0.4])
    assert result["drift_detected"] is False
    assert "warning" in result


def test_compute_psi_identical():
    from app.monitoring import compute_psi

    arr = np.random.default_rng(42).normal(0, 1, 500)
    psi = compute_psi(arr, arr.copy())
    assert psi < 0.1


def test_compute_psi_large_shift():
    from app.monitoring import compute_psi

    rng = np.random.default_rng(10)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(5, 1, 500)
    psi = compute_psi(ref, cur)
    assert psi > 0.2


@pytest.mark.parametrize("n_ref,n_cur", [(50, 50), (200, 100), (500, 200)])
def test_compute_drift_various_sizes(n_ref, n_cur):
    from app.monitoring import compute_drift

    rng = np.random.default_rng(99)
    ref = rng.normal(0, 1, n_ref).tolist()
    cur = rng.normal(0, 1, n_cur).tolist()
    result = compute_drift(ref, cur)
    assert 0.0 <= result["ks_statistic"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0


def test_check_feature_drift_returns_dict():
    from app.monitoring import check_feature_drift

    rng = np.random.default_rng(5)
    reference = [{"home_form": float(x), "away_form": float(y)} for x, y in zip(rng.normal(0.6, 0.1, 50), rng.normal(0.4, 0.1, 50))]
    current = [{"home_form": float(x), "away_form": float(y)} for x, y in zip(rng.normal(0.6, 0.1, 30), rng.normal(0.4, 0.1, 30))]
    results = check_feature_drift(reference, current)
    assert isinstance(results, dict)
    for v in results.values():
        assert "ks_statistic" in v


def test_check_feature_drift_empty():
    from app.monitoring import check_feature_drift

    result = check_feature_drift([], [])
    assert result == {}
