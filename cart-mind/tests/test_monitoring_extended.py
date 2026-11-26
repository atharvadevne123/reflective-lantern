"""Extended cart-mind monitoring tests with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.monitoring import (
    compute_drift,
    get_reference_window,
    update_reference_window,
)


def test_compute_drift_insufficient_reference_returns_reason() -> None:
    result = compute_drift([1.0] * 5, [1.0] * 50)
    assert result["drift_detected"] is False
    assert "reason" in result


def test_compute_drift_insufficient_current_returns_reason() -> None:
    result = compute_drift([1.0] * 50, [1.0] * 5)
    assert result["drift_detected"] is False
    assert "reason" in result


def test_compute_drift_both_insufficient_returns_reason() -> None:
    result = compute_drift([1.0] * 5, [1.0] * 5)
    assert "reason" in result


def test_compute_drift_identical_data_no_drift() -> None:
    data = [float(i) for i in range(100)]
    result = compute_drift(data, data)
    assert result["drift_detected"] is False
    assert result["ks_statistic"] == pytest.approx(0.0, abs=1e-6)


def test_compute_drift_very_different_data_drift() -> None:
    reference = [float(i) for i in range(100)]
    current = [float(i + 500) for i in range(100)]
    result = compute_drift(reference, current)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_returns_required_keys() -> None:
    data = [float(i) for i in range(50)]
    result = compute_drift(data, data)
    for key in ("ks_statistic", "p_value", "drift_detected"):
        assert key in result


@pytest.mark.parametrize("mean_shift", [20.0, 50.0, 100.0])
def test_compute_drift_large_shift_detected(mean_shift: float) -> None:
    reference = [float(i) for i in range(100)]
    current = [float(i) + mean_shift for i in range(100)]
    result = compute_drift(reference, current)
    assert result["drift_detected"] is True


def test_update_reference_window_stores_values() -> None:
    update_reference_window("test_feature_42", [1.0, 2.0, 3.0])
    window = get_reference_window("test_feature_42")
    assert all(v in window for v in [1.0, 2.0, 3.0])


def test_get_reference_window_missing_feature_returns_empty() -> None:
    window = get_reference_window("feature_that_does_not_exist_xyz")
    assert window == []


def test_update_reference_window_accumulates() -> None:
    feature_name = "accum_feature_test"
    update_reference_window(feature_name, [1.0, 2.0])
    update_reference_window(feature_name, [3.0, 4.0])
    window = get_reference_window(feature_name)
    assert len(window) >= 4


@pytest.mark.parametrize("n_values", [1, 10, 50, 100])
def test_update_reference_window_various_sizes(n_values: int) -> None:
    feature_name = f"test_feat_{n_values}"
    values = [float(i) for i in range(n_values)]
    update_reference_window(feature_name, values)
    window = get_reference_window(feature_name)
    assert len(window) > 0
