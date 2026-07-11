"""Tests for the synthetic data generator."""

from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from seed_data import generate_readings


class TestGenerateReadings:
    def test_count_matches(self):
        readings = generate_readings(n=100)
        assert len(readings) == 100

    def test_deterministic_with_seed(self):
        a = generate_readings(n=50, seed=7)
        b = generate_readings(n=50, seed=7)
        assert a == b

    def test_has_all_channels(self):
        readings = generate_readings(n=5)
        for r in readings:
            assert set(r["values"]) == {"temperature", "vibration", "rpm", "power_kw"}

    def test_anomaly_rate_approximate(self):
        readings = generate_readings(n=2000, anomaly_rate=0.10, seed=1)
        injected = sum(1 for r in readings if r["_injected_anomaly"])
        assert 120 < injected < 280  # ~10% of 2000, generous bounds

    def test_zero_anomaly_rate(self):
        readings = generate_readings(n=200, anomaly_rate=0.0)
        assert not any(r["_injected_anomaly"] for r in readings)

    def test_timestamps_monotonic(self):
        readings = generate_readings(n=20)
        timestamps = [r["timestamp"] for r in readings]
        assert timestamps == sorted(timestamps)

    def test_custom_sensor_id(self):
        readings = generate_readings(n=3, sensor_id="pump-99")
        assert all(r["sensor_id"] == "pump-99" for r in readings)

    def test_injected_anomalies_have_higher_vibration(self):
        readings = generate_readings(n=2000, anomaly_rate=0.05, seed=3)
        normal = [r["values"]["vibration"] for r in readings if not r["_injected_anomaly"]]
        anomalous = [r["values"]["vibration"] for r in readings if r["_injected_anomaly"]]
        assert sum(anomalous) / len(anomalous) > sum(normal) / len(normal)
