"""Extended monitoring tests: drift detection edge cases and parametrize."""
from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import compute_drift


@pytest.mark.parametrize("ref_size,cur_size", [
    (0, 0),
    (5, 5),
    (9, 10),
    (10, 9),
])
def test_compute_drift_insufficient_samples(ref_size: int, cur_size: int) -> None:
    ref = list(np.random.default_rng(1).uniform(0, 1, ref_size))
    cur = list(np.random.default_rng(2).uniform(0, 1, cur_size))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is False
    assert result.get("error") == "insufficient_samples"
    assert result["ks_statistic"] is None
    assert result["p_value"] is None


def test_compute_drift_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    data = list(rng.normal(0, 1, 100))
    result = compute_drift(data, data)
    assert result["drift_detected"] is False
    assert result["ks_statistic"] == pytest.approx(0.0, abs=1e-6)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-6)


def test_compute_drift_clearly_different_distributions() -> None:
    ref = list(np.random.default_rng(1).normal(0, 1, 200))
    cur = list(np.random.default_rng(2).normal(10, 1, 200))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5
    assert result["p_value"] < 0.05


def test_compute_drift_returns_rounded_values() -> None:
    ref = list(np.random.default_rng(10).normal(0, 1, 50))
    cur = list(np.random.default_rng(11).normal(1, 1, 50))
    result = compute_drift(ref, cur)
    if result["ks_statistic"] is not None:
        assert round(result["ks_statistic"], 4) == result["ks_statistic"]
        assert round(result["p_value"], 4) == result["p_value"]


def test_compute_drift_result_has_required_keys() -> None:
    ref = list(np.random.default_rng(5).normal(0, 1, 50))
    cur = list(np.random.default_rng(6).normal(0, 1, 50))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result


def test_compute_drift_boolean_flag() -> None:
    ref = list(np.random.default_rng(7).normal(0, 1, 50))
    cur = list(np.random.default_rng(8).normal(0, 1, 50))
    result = compute_drift(ref, cur)
    assert isinstance(result["drift_detected"], bool)


@pytest.mark.parametrize("mean_shift", [0.5, 1.0, 2.0, 3.0, 5.0])
def test_compute_drift_detects_at_large_shift(mean_shift: float) -> None:
    rng = np.random.default_rng(99)
    ref = list(rng.normal(0, 1, 500))
    cur = list(rng.normal(mean_shift, 1, 500))
    result = compute_drift(ref, cur)
    if mean_shift >= 2.0:
        assert result["drift_detected"] is True


def test_compute_drift_exact_minimum_samples() -> None:
    ref = list(np.arange(10, dtype=float))
    cur = list(np.arange(10, 20, dtype=float))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert result.get("error") is None


def test_compute_drift_ks_statistic_range() -> None:
    ref = list(np.random.default_rng(20).normal(0, 1, 100))
    cur = list(np.random.default_rng(21).normal(1, 1, 100))
    result = compute_drift(ref, cur)
    if result["ks_statistic"] is not None:
        assert 0.0 <= result["ks_statistic"] <= 1.0


def test_compute_drift_p_value_range() -> None:
    ref = list(np.random.default_rng(30).uniform(0, 1, 100))
    cur = list(np.random.default_rng(31).uniform(0, 1, 100))
    result = compute_drift(ref, cur)
    if result["p_value"] is not None:
        assert 0.0 <= result["p_value"] <= 1.0
