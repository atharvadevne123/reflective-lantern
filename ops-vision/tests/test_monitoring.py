"""Tests for Ops-Vision KS-test drift detection."""

import numpy as np
import pytest

from app.monitoring import FEATURE_COLS, DriftMonitor, DriftResult


def _make_sample(
    cpu: float = 40.0,
    mem: float = 50.0,
    err: float = 2.0,
    lat: float = 150.0,
    req: float = 200.0,
    disk: float = 30.0,
) -> dict:
    """Build a minimal metrics dict for drift tests."""
    return {
        "cpu_usage_pct": cpu,
        "memory_usage_pct": mem,
        "error_rate_per_min": err,
        "latency_p99_ms": lat,
        "request_rate_per_sec": req,
        "disk_io_util_pct": disk,
    }


def _normal_batch(n: int = 200, rng=None) -> list[dict]:
    """Generate n samples from the normal (non-incident) distribution."""
    if rng is None:
        rng = np.random.default_rng(0)
    return [
        _make_sample(
            cpu=float(rng.normal(40, 5)),
            mem=float(rng.normal(50, 5)),
            err=float(rng.exponential(2)),
            lat=float(rng.normal(150, 20)),
            req=float(rng.normal(200, 30)),
            disk=float(rng.normal(30, 5)),
        )
        for _ in range(n)
    ]


def _drifted_batch(n: int = 200, rng=None) -> list[dict]:
    """Generate n samples from the shifted (incident) distribution."""
    if rng is None:
        rng = np.random.default_rng(99)
    return [
        _make_sample(
            cpu=float(rng.normal(85, 5)),
            mem=float(rng.normal(88, 5)),
            err=float(rng.normal(60, 10)),
            lat=float(rng.normal(1500, 200)),
            req=float(rng.normal(50, 10)),
            disk=float(rng.normal(85, 5)),
        )
        for _ in range(n)
    ]


class TestDriftMonitorInit:
    """Tests for DriftMonitor initialisation."""

    def test_initial_window_sizes_are_zero(self):
        """New monitor has empty reference and current windows."""
        monitor = DriftMonitor()
        assert monitor.reference_size == 0
        assert monitor.current_size == 0

    def test_custom_threshold(self):
        """Monitor stores custom threshold."""
        monitor = DriftMonitor(threshold=0.01)
        assert monitor.threshold == 0.01


class TestUpdateReference:
    """Tests for update_reference()."""

    def test_reference_size_increases(self):
        """Reference window grows after update_reference call."""
        monitor = DriftMonitor()
        monitor.update_reference(_normal_batch(50))
        assert monitor.reference_size == 50

    def test_reference_capped_at_max(self):
        """Reference window is capped at reference_window_size."""
        monitor = DriftMonitor(reference_window_size=100)
        monitor.update_reference(_normal_batch(200))
        assert monitor.reference_size == 100


class TestRecord:
    """Tests for the record() streaming interface."""

    def test_record_returns_none_before_window_full(self):
        """record() returns None while current window is not full."""
        monitor = DriftMonitor(current_window_size=10)
        monitor.update_reference(_normal_batch(50))
        result = monitor.record(_make_sample())
        assert result is None

    def test_record_returns_results_when_window_full(self):
        """record() triggers a drift check when current window reaches capacity."""
        monitor = DriftMonitor(current_window_size=5)
        monitor.update_reference(_normal_batch(50))
        result = None
        for sample in _normal_batch(5):
            result = monitor.record(sample)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == len(FEATURE_COLS)

    def test_record_clears_window_after_check(self):
        """Current window is cleared after a drift check is triggered."""
        monitor = DriftMonitor(current_window_size=5)
        monitor.update_reference(_normal_batch(50))
        for sample in _normal_batch(5):
            monitor.record(sample)
        assert monitor.current_size == 0


class TestCheckDrift:
    """Tests for explicit check_drift() calls."""

    def test_drift_detected_on_shifted_distribution(self):
        """Drifted distribution should trigger at least one KS failure."""
        monitor = DriftMonitor(current_window_size=200)
        monitor.update_reference(_normal_batch(500))
        for s in _drifted_batch(200):
            monitor._current.append(s)
        results = monitor.check_drift()
        drifted = [r for r in results if r.drifted]
        assert len(drifted) > 0

    def test_no_drift_on_same_distribution(self):
        """Identical distribution should not trigger drift."""
        rng = np.random.default_rng(42)
        monitor = DriftMonitor(current_window_size=200)
        monitor.update_reference(_normal_batch(500, rng=rng))
        for s in _normal_batch(200, rng=rng):
            monitor._current.append(s)
        results = monitor.check_drift()
        drifted = [r for r in results if r.drifted]
        assert len(drifted) == 0

    def test_check_drift_raises_if_reference_empty(self):
        """check_drift raises ValueError when reference window is empty."""
        monitor = DriftMonitor()
        monitor._current.append(_make_sample())
        with pytest.raises(ValueError, match="Reference window is empty"):
            monitor.check_drift()

    def test_drift_result_fields(self):
        """DriftResult objects have the expected fields."""
        monitor = DriftMonitor(current_window_size=50)
        monitor.update_reference(_normal_batch(200))
        for s in _drifted_batch(50):
            monitor._current.append(s)
        results = monitor.check_drift()
        for r in results:
            assert isinstance(r, DriftResult)
            assert r.feature_name in FEATURE_COLS
            assert 0.0 <= r.ks_statistic <= 1.0
            assert 0.0 <= r.p_value <= 1.0
            assert isinstance(r.drifted, bool)

    @pytest.mark.parametrize("feature", FEATURE_COLS)
    def test_all_features_are_tested(self, feature):
        """check_drift returns one result per feature column."""
        monitor = DriftMonitor(current_window_size=50)
        monitor.update_reference(_normal_batch(200))
        for s in _normal_batch(50):
            monitor._current.append(s)
        results = monitor.check_drift()
        feature_names = {r.feature_name for r in results}
        assert feature in feature_names


class TestGetMonitorSingleton:
    """Tests for the module-level singleton helper."""

    def test_get_monitor_returns_drift_monitor(self):
        """get_monitor() returns a DriftMonitor instance."""
        from app.monitoring import get_monitor

        monitor = get_monitor()
        assert isinstance(monitor, DriftMonitor)

    def test_get_monitor_returns_same_instance(self):
        """get_monitor() returns the same singleton on repeated calls."""
        from app.monitoring import get_monitor

        m1 = get_monitor()
        m2 = get_monitor()
        assert m1 is m2


class TestDriftMonitorReset:
    """Tests for DriftMonitor.reset() and drifted_features()."""

    def test_reset_clears_reference_window(self):
        """reset() empties the reference window."""
        monitor = DriftMonitor()
        monitor.update_reference(_normal_batch(100))
        assert monitor.reference_size == 100
        monitor.reset()
        assert monitor.reference_size == 0

    def test_reset_clears_current_window(self):
        """reset() empties the current window."""
        monitor = DriftMonitor(current_window_size=50)
        monitor.update_reference(_normal_batch(50))
        for s in _normal_batch(30):
            monitor._current.append(s)
        monitor.reset()
        assert monitor.current_size == 0

    def test_drifted_features_returns_correct_names(self):
        """drifted_features() filters DriftResult list to drifted names."""
        monitor = DriftMonitor(current_window_size=50)
        monitor.update_reference(_normal_batch(200))
        for s in _drifted_batch(50):
            monitor._current.append(s)
        results = monitor.check_drift()
        drifted = monitor.drifted_features(results)
        assert isinstance(drifted, list)
        assert all(name in FEATURE_COLS for name in drifted)

    def test_drifted_features_empty_when_no_drift(self):
        """drifted_features() returns [] when no features drifted."""
        monitor = DriftMonitor(current_window_size=50)
        rng = np.random.default_rng(7)
        monitor.update_reference(_normal_batch(200, rng=rng))
        for s in _normal_batch(50, rng=rng):
            monitor._current.append(s)
        results = monitor.check_drift()
        stable_only = [r for r in results if not r.drifted]
        assert monitor.drifted_features(stable_only) == []


class TestDriftResultRepr:
    """Tests for DriftResult.__repr__."""

    def test_repr_contains_feature_name(self):
        """DriftResult repr includes the feature name."""
        r = DriftResult(
            feature_name="cpu_usage_pct",
            ks_statistic=0.3,
            p_value=0.001,
            drifted=True,
        )
        assert "cpu_usage_pct" in repr(r)

    def test_repr_contains_drifted_status(self):
        """DriftResult repr says DRIFTED for a drifted result."""
        r = DriftResult(
            feature_name="latency_p99_ms",
            ks_statistic=0.45,
            p_value=0.0001,
            drifted=True,
        )
        assert "DRIFTED" in repr(r)

    def test_repr_contains_stable_status(self):
        """DriftResult repr says stable for a non-drifted result."""
        r = DriftResult(
            feature_name="memory_usage_pct",
            ks_statistic=0.05,
            p_value=0.8,
            drifted=False,
        )
        assert "stable" in repr(r)


class TestDriftMonitorSummary:
    """Tests for DriftMonitor.summary()."""

    def test_summary_returns_dict(self):
        """summary() returns a dict."""
        monitor = DriftMonitor()
        result = monitor.summary()
        assert isinstance(result, dict)

    def test_summary_has_reference_size(self):
        """summary() dict contains reference_size key."""
        monitor = DriftMonitor()
        assert "reference_size" in monitor.summary()

    def test_summary_has_current_size(self):
        """summary() dict contains current_size key."""
        monitor = DriftMonitor()
        assert "current_size" in monitor.summary()

    def test_summary_reference_size_matches(self):
        """summary reference_size matches actual reference window size."""
        monitor = DriftMonitor()
        monitor.update_reference(_normal_batch(50))
        assert monitor.summary()["reference_size"] == 50

    def test_summary_feature_count_correct(self):
        """summary feature_count equals FEATURE_COLS length."""
        monitor = DriftMonitor()
        assert monitor.summary()["feature_count"] == len(FEATURE_COLS)

    def test_summary_threshold_matches_init(self):
        """summary threshold matches the value passed at construction."""
        monitor = DriftMonitor(threshold=0.01)
        assert monitor.summary()["threshold"] == 0.01
