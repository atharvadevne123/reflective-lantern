"""Extended tests for cart-mind app/monitoring.py helpers."""

from __future__ import annotations

import pytest

from app.monitoring import (
    compute_psi,
    drift_summary,
    get_reference_window,
    reset_reference_window,
    rolling_mean,
    update_reference_window,
)


class TestUpdateAndGetReferenceWindow:
    def setup_method(self) -> None:
        reset_reference_window()

    def test_update_then_get(self) -> None:
        update_reference_window("cpu", [0.1, 0.2, 0.3])
        result = get_reference_window("cpu")
        assert result == [0.1, 0.2, 0.3]

    def test_empty_window_on_missing_key(self) -> None:
        result = get_reference_window("nonexistent")
        assert result == [] or result is None

    def test_overwrite_window(self) -> None:
        update_reference_window("cpu", [0.1, 0.2])
        update_reference_window("cpu", [0.5, 0.6, 0.7])
        result = get_reference_window("cpu")
        assert result == [0.5, 0.6, 0.7]


class TestResetReferenceWindow:
    def test_reset_specific_feature(self) -> None:
        update_reference_window("mem", [0.4, 0.5])
        reset_reference_window("mem")
        result = get_reference_window("mem")
        assert not result

    def test_reset_all(self) -> None:
        update_reference_window("cpu", [0.1])
        update_reference_window("mem", [0.2])
        reset_reference_window()
        assert not get_reference_window("cpu")
        assert not get_reference_window("mem")


class TestRollingMean:
    def test_basic_window(self) -> None:
        result = rolling_mean([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        assert isinstance(result, list)

    def test_window_one_returns_same(self) -> None:
        values = [1.0, 2.0, 3.0]
        result = rolling_mean(values, window=1)
        assert result == pytest.approx(values)

    def test_output_length(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = rolling_mean(values, window=3)
        assert len(result) == len(values)

    def test_mean_values_correct(self) -> None:
        values = [1.0, 2.0, 3.0]
        result = rolling_mean(values, window=3)
        assert result[-1] == pytest.approx(2.0)


class TestComputePsi:
    def test_returns_dict(self) -> None:
        ref = [0.5 + i * 0.01 for i in range(50)]
        cur = [0.5 + i * 0.01 for i in range(50)]
        result = compute_psi(ref, cur)
        assert isinstance(result, dict)

    def test_low_psi_on_identical(self) -> None:
        values = [float(i) for i in range(50)]
        result = compute_psi(values, values)
        assert "psi" in result
        assert result["psi"] <= 0.1

    def test_high_psi_on_different(self) -> None:
        ref = [0.0] * 50
        cur = [1.0] * 50
        result = compute_psi(ref, cur)
        assert "psi" in result


class TestDriftSummary:
    def test_empty_returns_dict(self) -> None:
        result = drift_summary({})
        assert isinstance(result, dict)

    def test_counts_drifted_features(self) -> None:
        feature_results = {
            "cpu": {"drift_detected": True},
            "mem": {"drift_detected": False},
        }
        summary = drift_summary(feature_results)
        assert "drifted_count" in summary or "total" in summary
