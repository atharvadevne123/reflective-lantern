"""Tests for fleet-mind app/data_drift.py."""

from __future__ import annotations

import pytest

from app.data_drift import (
    DriftResult,
    batch_drift,
    detect_drift,
    ks_statistic,
    z_score_shift,
)


class TestKsStatistic:
    def test_identical_distributions_zero(self) -> None:
        values = [float(i) for i in range(50)]
        stat = ks_statistic(values, values)
        assert stat == pytest.approx(0.0)

    def test_disjoint_distributions_high(self) -> None:
        ref = [0.0] * 50
        cur = [100.0] * 50
        stat = ks_statistic(ref, cur)
        assert stat > 0.5

    def test_empty_inputs_return_zero(self) -> None:
        assert ks_statistic([], []) == pytest.approx(0.0)


class TestZScoreShift:
    def test_no_shift(self) -> None:
        values = [float(i) for i in range(50)]
        assert z_score_shift(values, values) == pytest.approx(0.0, abs=0.01)

    def test_large_shift(self) -> None:
        ref = [0.0] * 50
        cur = [100.0] * 50
        shift = z_score_shift(ref, cur)
        assert abs(shift) > 5.0

    def test_too_few_reference_returns_zero(self) -> None:
        assert z_score_shift([5.0], [5.0]) == pytest.approx(0.0)


class TestDetectDrift:
    def test_no_drift_on_same_data(self) -> None:
        values = [float(i) for i in range(100)]
        result = detect_drift("temp", values, values)
        assert isinstance(result, DriftResult)
        assert not result.drift_detected

    def test_drift_detected_on_shifted_data(self) -> None:
        ref = [0.0] * 100
        cur = [100.0] * 100
        result = detect_drift("temp", ref, cur, z_threshold=2.0)
        assert result.drift_detected

    def test_result_contains_feature_name(self) -> None:
        values = [1.0] * 10
        result = detect_drift("engine_temp", values, values)
        assert result.feature == "engine_temp"


class TestBatchDrift:
    def test_returns_list_of_results(self) -> None:
        features = {
            "temp": ([90.0] * 20, [90.0] * 20),
            "pressure": ([3.0] * 20, [3.0] * 20),
        }
        results = batch_drift(features)
        assert len(results) == 2

    def test_result_types(self) -> None:
        features = {"x": ([1.0] * 10, [1.0] * 10)}
        results = batch_drift(features)
        assert all(isinstance(r, DriftResult) for r in results)
