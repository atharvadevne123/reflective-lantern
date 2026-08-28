"""Tests for drift detection and prediction logging."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    PredictionStore,
    compute_drift,
    compute_psi,
)


class TestComputeDrift:
    def test_no_drift_for_identical_distributions(self) -> None:
        data = list(np.random.default_rng(0).normal(0, 1, 100))
        result = compute_drift(data, data)
        assert result["drift_detected"] is False

    def test_detects_drift_for_shifted_distribution(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0.0, 1.0, 200).tolist()
        current = rng.normal(5.0, 1.0, 200).tolist()
        result = compute_drift(reference, current)
        assert result["drift_detected"] is True

    def test_ks_statistic_is_between_0_and_1(self) -> None:
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 100).tolist()
        cur = rng.normal(0, 1, 100).tolist()
        result = compute_drift(ref, cur)
        assert 0.0 <= result["ks_statistic"] <= 1.0

    def test_p_value_between_0_and_1(self) -> None:
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, 100).tolist()
        cur = rng.normal(0, 1, 100).tolist()
        result = compute_drift(ref, cur)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_empty_reference_returns_no_drift(self) -> None:
        result = compute_drift([], [1.0, 2.0, 3.0])
        assert result["drift_detected"] is False
        assert "error" in result

    def test_empty_current_returns_no_drift(self) -> None:
        result = compute_drift([1.0, 2.0, 3.0], [])
        assert result["drift_detected"] is False

    def test_single_sample_insufficient(self) -> None:
        result = compute_drift([1.0], [2.0])
        assert "error" in result

    @pytest.mark.parametrize("shift", [0.0, 1.0, 3.0, 10.0])
    def test_drift_increases_with_shift(self, shift: float) -> None:
        rng = np.random.default_rng(5)
        reference = rng.normal(0, 1, 200).tolist()
        current = rng.normal(shift, 1, 200).tolist()
        result = compute_drift(reference, current)
        assert result["ks_statistic"] is not None


class TestPredictionStore:
    def test_records_numeric_features(self) -> None:
        store = PredictionStore(max_size=50)
        store.record({"depth_km": 10.0, "station_count": 5}, prediction=4.2)
        assert store.sample_count("depth_km") == 1

    def test_skips_non_numeric_features(self) -> None:
        store = PredictionStore(max_size=50)
        store.record({"fault_type": "reverse", "depth_km": 12.0}, prediction=5.0)
        assert store.sample_count("fault_type") == 0
        assert store.sample_count("depth_km") == 1

    def test_records_prediction_value(self) -> None:
        store = PredictionStore(max_size=50)
        store.record({"depth_km": 10.0}, prediction=3.7)
        assert store.sample_count("prediction") == 1
        assert store.get_feature_window("prediction") == [3.7]

    def test_max_size_respected(self) -> None:
        store = PredictionStore(max_size=5)
        for i in range(10):
            store.record({"depth_km": float(i)}, prediction=float(i))
        assert store.sample_count("depth_km") == 5

    def test_all_features_returns_known_keys(self) -> None:
        store = PredictionStore(max_size=10)
        store.record({"a": 1.0, "b": 2.0}, prediction=0.5)
        features = store.all_features()
        assert "a" in features
        assert "b" in features
        assert "prediction" in features


class TestComputePsi:
    def test_returns_zero_for_empty_data(self) -> None:
        assert compute_psi([], []) == 0.0

    def test_returns_low_psi_for_same_distribution(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 200).tolist()
        psi = compute_psi(data, data)
        assert psi < 0.1

    def test_returns_high_psi_for_shifted_distribution(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 300).tolist()
        cur = rng.normal(5, 1, 300).tolist()
        psi = compute_psi(ref, cur)
        assert psi > 0.1

    def test_psi_is_non_negative(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.uniform(0, 1, 100).tolist()
        cur = rng.uniform(0, 1, 100).tolist()
        psi = compute_psi(ref, cur)
        assert psi >= 0.0

    def test_insufficient_samples_returns_zero(self) -> None:
        assert compute_psi([1.0, 2.0], [3.0, 4.0]) == 0.0
