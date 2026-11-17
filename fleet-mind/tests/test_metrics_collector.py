"""Tests for Fleet-Mind metrics collector."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.metrics_collector as mc_module
from app.metrics_collector import get_in_memory_metrics, record_drift, record_prediction


@pytest.fixture(autouse=True)
def reset_metrics():
    mc_module._counters.clear()
    mc_module._latencies.clear()
    yield
    mc_module._counters.clear()
    mc_module._latencies.clear()


def test_record_prediction_maintenance_increments_counter():
    record_prediction(maintenance_needed=True, latency_s=0.05)
    metrics = get_in_memory_metrics()
    assert metrics.get("predictions_maintenance", 0) >= 1


def test_record_prediction_normal_increments_counter():
    record_prediction(maintenance_needed=False, latency_s=0.02)
    metrics = get_in_memory_metrics()
    assert metrics.get("predictions_normal", 0) >= 1


def test_record_prediction_stores_latency():
    record_prediction(maintenance_needed=False, latency_s=0.1)
    metrics = get_in_memory_metrics()
    assert metrics["avg_latency_s"] > 0


def test_record_drift_increments_counter():
    record_drift("engine_temp_c")
    metrics = get_in_memory_metrics()
    assert metrics.get("drift_engine_temp_c", 0) >= 1


def test_get_in_memory_metrics_empty():
    metrics = get_in_memory_metrics()
    assert isinstance(metrics, dict)
    assert metrics["avg_latency_s"] == 0.0


def test_avg_latency_computed_correctly():
    record_prediction(maintenance_needed=False, latency_s=0.1)
    record_prediction(maintenance_needed=True, latency_s=0.3)
    metrics = get_in_memory_metrics()
    assert abs(metrics["avg_latency_s"] - 0.2) < 0.01


def test_multiple_drifts_accumulate():
    record_drift("rpm")
    record_drift("rpm")
    metrics = get_in_memory_metrics()
    assert metrics.get("drift_rpm", 0) >= 2


@pytest.mark.parametrize("maintenance", [True, False])
def test_prediction_counters_independent(maintenance):
    record_prediction(maintenance_needed=maintenance, latency_s=0.01)
    metrics = get_in_memory_metrics()
    key = "predictions_maintenance" if maintenance else "predictions_normal"
    assert metrics.get(key, 0) >= 1
