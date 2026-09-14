"""Tests for app.metrics_collector."""

import threading

import pytest

from app.metrics_collector import Counter, Gauge, Histogram, MetricsRegistry


class TestCounter:
    def test_initial_value_zero(self) -> None:
        c = Counter("req")
        assert c.value == 0.0

    def test_inc_default(self) -> None:
        c = Counter("req")
        c.inc()
        assert c.value == 1.0

    def test_inc_by_amount(self) -> None:
        c = Counter("bytes")
        c.inc(512)
        c.inc(512)
        assert c.value == 1024.0

    def test_negative_raises(self) -> None:
        c = Counter("x")
        with pytest.raises(ValueError):
            c.inc(-1)

    def test_reset(self) -> None:
        c = Counter("x")
        c.inc(10)
        c.reset()
        assert c.value == 0.0

    def test_thread_safe(self) -> None:
        c = Counter("t")
        threads = [threading.Thread(target=lambda: c.inc()) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.value == 100.0


class TestGauge:
    def test_set(self) -> None:
        g = Gauge("temp")
        g.set(37.5)
        assert g.value == 37.5

    def test_inc_dec(self) -> None:
        g = Gauge("workers")
        g.inc(5)
        g.dec(2)
        assert g.value == 3.0

    def test_negative_allowed(self) -> None:
        g = Gauge("delta")
        g.dec(10)
        assert g.value == -10.0


class TestHistogram:
    def test_observe_increments_count(self) -> None:
        h = Histogram("latency", buckets=[0.1, 0.5, 1.0])
        h.observe(0.05)
        h.observe(0.3)
        assert h.count == 2

    def test_sum_accumulates(self) -> None:
        h = Histogram("latency", buckets=[1.0])
        h.observe(0.4)
        h.observe(0.6)
        assert abs(h.sum - 1.0) < 1e-9

    def test_percentile_empty(self) -> None:
        h = Histogram("lat", buckets=[1.0])
        assert h.percentile(0.99) is None

    def test_percentile_estimate(self) -> None:
        h = Histogram("lat", buckets=[0.1, 0.5, 1.0])
        for _ in range(100):
            h.observe(0.05)
        p99 = h.percentile(0.99)
        assert p99 == 0.1


class TestMetricsRegistry:
    def test_counter_idempotent(self) -> None:
        reg = MetricsRegistry()
        c1 = reg.counter("hits")
        c2 = reg.counter("hits")
        assert c1 is c2

    def test_gauge_idempotent(self) -> None:
        reg = MetricsRegistry()
        assert reg.gauge("cpu") is reg.gauge("cpu")

    def test_histogram_idempotent(self) -> None:
        reg = MetricsRegistry()
        assert reg.histogram("lat") is reg.histogram("lat")

    def test_all_metrics_returns_dict(self) -> None:
        reg = MetricsRegistry()
        reg.counter("a")
        reg.gauge("b")
        metrics = reg.all_metrics()
        assert "a" in metrics
        assert "b" in metrics


class TestCounterEdgeCases:
    def test_inc_zero_is_valid(self) -> None:
        c = Counter("x")
        c.inc(0)
        assert c.value == 0.0

    def test_inc_fractional_amount(self) -> None:
        c = Counter("bytes")
        c.inc(0.5)
        c.inc(0.5)
        assert c.value == pytest.approx(1.0)

    def test_multiple_resets(self) -> None:
        c = Counter("x")
        for _ in range(3):
            c.inc(10)
            c.reset()
        assert c.value == 0.0

    @pytest.mark.parametrize("amount", [0.001, 1.0, 1_000_000.0])
    def test_various_amounts(self, amount: float) -> None:
        c = Counter("x")
        c.inc(amount)
        assert c.value == pytest.approx(amount)


class TestGaugeEdgeCases:
    def test_initial_value_is_zero(self) -> None:
        g = Gauge("x")
        assert g.value == 0.0

    def test_overwrite_with_set(self) -> None:
        g = Gauge("x")
        g.inc(100)
        g.set(5.0)
        assert g.value == 5.0

    def test_inc_dec_cancel(self) -> None:
        g = Gauge("x")
        g.inc(10)
        g.dec(10)
        assert g.value == pytest.approx(0.0)


class TestHistogramEdgeCases:
    def test_value_above_all_buckets_still_counted(self) -> None:
        h = Histogram("lat", buckets=[0.1, 0.5])
        h.observe(999.0)
        assert h.count == 1
        assert h.sum == pytest.approx(999.0)

    def test_percentile_when_all_in_first_bucket(self) -> None:
        h = Histogram("lat", buckets=[0.01, 0.1, 1.0])
        for _ in range(10):
            h.observe(0.001)
        assert h.percentile(0.5) == 0.01

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_count_matches_observations(self, n: int) -> None:
        h = Histogram("lat")
        for _ in range(n):
            h.observe(0.5)
        assert h.count == n


@pytest.mark.parametrize("n", [1, 5, 10])
def test_counter_increments_by_n(n: int) -> None:
    """Counter.inc(n) increases value by exactly n."""
    c = Counter("counter")
    c.inc(n)
    assert c.value == pytest.approx(float(n))


@pytest.mark.parametrize("value", [0.0, 5.0, 100.0])
def test_gauge_set_and_get(value: float) -> None:
    """Gauge.set() stores the exact value."""
    g = Gauge("gauge")
    g.set(value)
    assert g.value == pytest.approx(value)


class TestMetricsRegistryEdgeCases:
    def test_registry_get_returns_none_for_unknown(self) -> None:
        reg = MetricsRegistry()
        assert reg.get("unknown") is None

    def test_counter_registered_is_retrievable(self) -> None:
        reg = MetricsRegistry()
        c = Counter("requests")
        reg.register(c)
        assert reg.get("requests") is c

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_registry_names_count(self, n: int) -> None:
        reg = MetricsRegistry()
        for i in range(n):
            reg.register(Counter(f"metric_{i}"))
        assert len(reg.names()) == n
