"""Extended monitoring/drift tests with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.monitoring import compute_drift


def test_compute_drift_insufficient_reference_returns_error() -> None:
    result = compute_drift([1.0], [1.0, 2.0, 3.0])
    assert "error" in result
    assert result["drift_detected"] is False


def test_compute_drift_insufficient_current_returns_error() -> None:
    result = compute_drift([1.0, 2.0, 3.0], [1.0])
    assert "error" in result
    assert result["drift_detected"] is False


def test_compute_drift_both_empty_returns_error() -> None:
    result = compute_drift([], [])
    assert "error" in result


def test_compute_drift_identical_distributions_no_drift() -> None:
    data = list(range(100))
    result = compute_drift(data, data)
    assert result["drift_detected"] is False
    assert result["ks_statistic"] == pytest.approx(0.0, abs=1e-6)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-6)


def test_compute_drift_very_different_distributions_drift() -> None:
    reference = [float(i) for i in range(100)]
    current = [float(i + 1000) for i in range(100)]
    result = compute_drift(reference, current)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_returns_required_keys() -> None:
    reference = [float(i) for i in range(50)]
    current = [float(i) for i in range(50)]
    result = compute_drift(reference, current)
    for key in ("ks_statistic", "p_value", "drift_detected"):
        assert key in result


def test_compute_drift_ks_statistic_in_zero_one() -> None:
    reference = [float(i) for i in range(100)]
    current = [float(i * 2) for i in range(100)]
    result = compute_drift(reference, current)
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_compute_drift_p_value_in_zero_one() -> None:
    reference = [float(i) for i in range(100)]
    current = [float(i + 10) for i in range(100)]
    result = compute_drift(reference, current)
    assert 0.0 <= result["p_value"] <= 1.0


def test_compute_drift_drift_detected_is_bool() -> None:
    reference = [float(i) for i in range(100)]
    current = [float(i) for i in range(100)]
    result = compute_drift(reference, current)
    assert isinstance(result["drift_detected"], bool)


@pytest.mark.parametrize("mean_shift", [10.0, 50.0, 100.0, 500.0])
def test_compute_drift_large_mean_shift_detects_drift(mean_shift: float) -> None:
    reference = [float(i) for i in range(100)]
    current = [float(i) + mean_shift for i in range(100)]
    result = compute_drift(reference, current)
    assert result["drift_detected"] is True


@pytest.mark.parametrize("n", [2, 5, 10, 50, 100])
def test_compute_drift_various_sample_sizes_no_error(n: int) -> None:
    reference = [float(i) for i in range(n)]
    current = [float(i) for i in range(n)]
    result = compute_drift(reference, current)
    assert "error" not in result


def test_compute_drift_similar_distributions_low_statistic() -> None:
    import random
    rng = random.Random(42)
    reference = [rng.gauss(0, 1) for _ in range(200)]
    current = [rng.gauss(0, 1) for _ in range(200)]
    result = compute_drift(reference, current)
    assert result["ks_statistic"] < 0.3
