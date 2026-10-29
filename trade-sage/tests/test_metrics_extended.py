"""Extended tests for service metrics counters and reset utility."""

from __future__ import annotations

import pytest

from app.metrics import ServiceMetrics, record_prediction_latency, reset_metrics


@pytest.fixture(autouse=True)
def restore_metrics():
    """Reset global metrics before and after each test."""
    reset_metrics()
    yield
    reset_metrics()


class TestServiceMetrics:
    def test_initial_counts_zero(self) -> None:
        m = ServiceMetrics()
        assert m.prediction_count == 0
        assert m.drift_alert_count == 0
        assert m.error_count == 0

    def test_increment_predictions(self) -> None:
        m = ServiceMetrics()
        m.increment_predictions()
        m.increment_predictions()
        assert m.prediction_count == 2

    def test_increment_drift_alerts_by_n(self) -> None:
        m = ServiceMetrics()
        m.increment_drift_alerts(n=5)
        assert m.drift_alert_count == 5

    def test_increment_errors(self) -> None:
        m = ServiceMetrics()
        m.increment_errors()
        assert m.error_count == 1

    def test_cache_hit_rate_zero_when_no_requests(self) -> None:
        m = ServiceMetrics()
        assert m.cache_hit_rate == 0.0

    def test_cache_hit_rate_correct(self) -> None:
        m = ServiceMetrics()
        m.record_cache_hit()
        m.record_cache_hit()
        m.record_cache_miss()
        assert abs(m.cache_hit_rate - 2 / 3) < 1e-6

    def test_snapshot_keys(self) -> None:
        m = ServiceMetrics()
        s = m.snapshot()
        expected = {"uptime_seconds", "prediction_count", "drift_alert_count",
                    "error_count", "cache_hits", "cache_misses", "cache_hit_rate"}
        assert expected.issubset(set(s.keys()))

    def test_uptime_non_negative(self) -> None:
        m = ServiceMetrics()
        assert m.uptime_seconds >= 0.0


class TestResetMetrics:
    def test_resets_prediction_count(self) -> None:
        from app.metrics import _service_metrics as sm
        sm.increment_predictions()
        reset_metrics()
        from app.metrics import _service_metrics as sm2
        assert sm2.prediction_count == 0

    def test_creates_new_instance(self) -> None:
        from app.metrics import _service_metrics as sm1
        reset_metrics()
        from app.metrics import _service_metrics as sm2
        assert sm1 is not sm2


class TestRecordPredictionLatency:
    def test_does_not_raise(self) -> None:
        record_prediction_latency(12.5)

    @pytest.mark.parametrize("latency", [0.0, 0.1, 100.0, 5000.0])
    def test_various_latencies(self, latency: float) -> None:
        record_prediction_latency(latency)
