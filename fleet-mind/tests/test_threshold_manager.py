"""Tests for fleet-mind app/threshold_manager.py."""

from __future__ import annotations

import pytest

from app.threshold_manager import (
    Threshold,
    ThresholdStore,
    adaptive_thresholds,
    batch_evaluate,
    evaluate,
)


class TestEvaluate:
    def test_ok_below_warning(self) -> None:
        t = Threshold("temp", warning=100.0, critical=120.0)
        assert evaluate(80.0, t) == "ok"

    def test_warning_at_threshold(self) -> None:
        t = Threshold("temp", warning=100.0, critical=120.0)
        assert evaluate(100.0, t) == "warning"

    def test_critical_at_threshold(self) -> None:
        t = Threshold("temp", warning=100.0, critical=120.0)
        assert evaluate(120.0, t) == "critical"

    def test_low_direction_ok(self) -> None:
        t = Threshold("pressure", warning=2.0, critical=1.0, direction="low")
        assert evaluate(5.0, t) == "ok"

    def test_low_direction_critical(self) -> None:
        t = Threshold("pressure", warning=2.0, critical=1.0, direction="low")
        assert evaluate(0.5, t) == "critical"


class TestAdaptiveThresholds:
    def test_returns_two_values(self) -> None:
        data = [float(i) for i in range(50)]
        warn, crit = adaptive_thresholds(data)
        assert warn < crit

    def test_too_few_values_raises(self) -> None:
        with pytest.raises(ValueError):
            adaptive_thresholds([5.0])


class TestThresholdStore:
    def test_set_and_get(self) -> None:
        store = ThresholdStore()
        t = Threshold("cpu", 70.0, 90.0)
        store.set(t)
        assert store.get("cpu") is t

    def test_missing_returns_none(self) -> None:
        store = ThresholdStore()
        assert store.get("missing") is None


class TestBatchEvaluate:
    def test_known_metric_evaluated(self) -> None:
        store = ThresholdStore()
        store.set(Threshold("temp", 100.0, 120.0))
        result = batch_evaluate({"temp": 80.0}, store)
        assert result["temp"] == "ok"

    def test_unknown_metric_is_unknown(self) -> None:
        store = ThresholdStore()
        result = batch_evaluate({"unknown": 50.0}, store)
        assert result["unknown"] == "unknown"
