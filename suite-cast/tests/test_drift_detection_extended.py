"""Extended drift detection tests with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.monitoring import compute_drift


@pytest.mark.parametrize("ref_size,cur_size", [(0, 10), (4, 10), (10, 0), (10, 3)])
def test_compute_drift_insufficient_data_returns_reason(ref_size: int, cur_size: int) -> None:
    reference = [float(i) for i in range(ref_size)]
    current = [float(i) for i in range(cur_size)]
    result = compute_drift(reference, current)
    assert "reason" in result
    assert result["drift_detected"] is False


def test_compute_drift_identical_no_drift() -> None:
    data = [float(i) for i in range(50)]
    result = compute_drift(data, data)
    assert result["drift_detected"] is False
    assert result["ks_statistic"] == pytest.approx(0.0, abs=1e-6)


def test_compute_drift_very_different_distributions_drift() -> None:
    reference = [float(i) for i in range(50)]
    current = [float(i + 500) for i in range(50)]
    result = compute_drift(reference, current)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_returns_required_keys() -> None:
    data = [float(i) for i in range(50)]
    result = compute_drift(data, data)
    for key in ("ks_statistic", "p_value", "drift_detected"):
        assert key in result


def test_compute_drift_ks_statistic_in_zero_one() -> None:
    reference = [float(i) for i in range(50)]
    current = [float(i * 2) for i in range(50)]
    result = compute_drift(reference, current)
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_compute_drift_p_value_in_zero_one() -> None:
    reference = [float(i) for i in range(50)]
    current = [float(i + 10) for i in range(50)]
    result = compute_drift(reference, current)
    assert 0.0 <= result["p_value"] <= 1.0


def test_compute_drift_drift_detected_is_bool() -> None:
    reference = [float(i) for i in range(50)]
    current = [float(i) for i in range(50)]
    result = compute_drift(reference, current)
    assert isinstance(result["drift_detected"], bool)


@pytest.mark.parametrize("mean_shift", [20.0, 50.0, 100.0])
def test_compute_drift_large_shift_detects_drift(mean_shift: float) -> None:
    reference = [float(i) for i in range(50)]
    current = [float(i) + mean_shift for i in range(50)]
    result = compute_drift(reference, current)
    assert result["drift_detected"] is True


@pytest.mark.parametrize("n", [5, 10, 20, 50])
def test_compute_drift_various_sample_sizes(n: int) -> None:
    reference = [float(i) for i in range(n)]
    current = [float(i) for i in range(n)]
    result = compute_drift(reference, current)
    assert "drift_detected" in result
    assert "ks_statistic" in result
