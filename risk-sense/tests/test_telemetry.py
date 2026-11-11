"""Telemetry tests for Risk-Sense."""

from __future__ import annotations

import pytest

from app.telemetry import (
    get_counter,
    get_gauge,
    increment,
    record_prediction,
    reset_all,
    set_gauge,
    snapshot,
)


@pytest.fixture(autouse=True)
def reset_telemetry() -> None:
    reset_all()


class TestCounters:
    def test_increment_starts_at_zero(self) -> None:
        assert get_counter("test_key") == 0

    def test_increment_adds_one(self) -> None:
        increment("test_key")
        assert get_counter("test_key") == 1

    def test_increment_multiple_times(self) -> None:
        for _ in range(5):
            increment("test_key")
        assert get_counter("test_key") == 5

    def test_increment_by_custom_amount(self) -> None:
        increment("test_key", amount=10)
        assert get_counter("test_key") == 10

    def test_different_keys_independent(self) -> None:
        increment("key_a")
        increment("key_b", amount=3)
        assert get_counter("key_a") == 1
        assert get_counter("key_b") == 3


class TestGauges:
    def test_gauge_defaults_to_zero(self) -> None:
        assert get_gauge("test_gauge") == 0.0

    def test_set_gauge(self) -> None:
        set_gauge("test_gauge", 42.0)
        assert get_gauge("test_gauge") == 42.0

    def test_gauge_overwrites(self) -> None:
        set_gauge("test_gauge", 10.0)
        set_gauge("test_gauge", 20.0)
        assert get_gauge("test_gauge") == 20.0


class TestSnapshot:
    def test_snapshot_contains_counters_and_gauges(self) -> None:
        increment("c1")
        set_gauge("g1", 1.5)
        result = snapshot()
        assert "counters" in result
        assert "gauges" in result
        assert result["counters"]["c1"] == 1
        assert result["gauges"]["g1"] == 1.5

    def test_snapshot_is_copy(self) -> None:
        s = snapshot()
        s["counters"]["mutated"] = 999
        assert get_counter("mutated") == 0


class TestRecordPrediction:
    def test_increments_total(self) -> None:
        record_prediction(0.3)
        assert get_counter("predictions_total") == 1

    def test_high_risk_incremented_for_high_prob(self) -> None:
        record_prediction(0.8)
        assert get_counter("high_risk_predictions") == 1

    def test_low_risk_incremented_for_low_prob(self) -> None:
        record_prediction(0.1)
        assert get_counter("low_risk_predictions") == 1

    @pytest.mark.parametrize("prob", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_various_probabilities_increment_total(self, prob: float) -> None:
        record_prediction(prob)
        assert get_counter("predictions_total") == 1
