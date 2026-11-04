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

    def test_initial_window_sizes_are_zero(self) -> None:
        """New monitor has empty reference and current windows."""
        monitor = DriftMonitor()
        assert monitor.reference_size == 0
        assert monitor.current_size == 0

    def test_custom_threshold(self) -> None:
        """Monitor stores custom threshold."""
        monitor = DriftMonitor(threshold=0.01)
        assert monitor.threshold == 0.01


class TestUpdateReference:
    """Tests for update_reference()."""

    def test_reference_size_increases(self) -> None:
        """Reference window grows after update_reference call."""
        monitor = DriftMonitor()
        monitor.update_reference(_normal_batch(50))
        assert monitor.reference_size == 50

    def test_reference_capped_at_max(self) -> None:
        """Reference window is capped at reference_window_size."""
        monitor = DriftMonitor(reference_window_size=100)
        monitor.update_reference(_normal_batch(200))
        assert monitor.reference_size == 100


class TestRecord:
    """Tests for the record() streaming interface."""

    def test_record_returns_none_before_window_full(self) -> None:
        """record() returns None while current window is not full."""
        monitor = DriftMonitor(current_window_size=10)
        monitor.update_reference(_normal_batch(50))
        result = monitor.record(_make_sample())
        assert result is None

    def test_record_returns_results_when_window_full(self) -> None:
        """record() triggers a drift check when current window reaches capacity."""
        monitor = DriftMonitor(current_window_size=5)
        monitor.update_reference(_normal_batch(50))
        result = None
        for sample in _normal_batch(5):
            result = monitor.record(sample)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == len(FEATURE_COLS)

    def test_record_clears_window_after_check(self) -> None:
        """Current window is cleared after a drift check is triggered."""
        monitor = DriftMonitor(current_window_size=5)
        monitor.update_reference(_normal_batch(50))
        for sample in _normal_batch(5):
            monitor.record(sample)
        assert monitor.current_size == 0


class TestCheckDrift:
    """Tests for explicit check_drift() calls."""

    def test_drift_detected_on_shifted_distribution(self) -> None:
        """Drifted distribution should trigger at least one KS failure."""
        monitor = DriftMonitor(current_window_size=200)
        monitor.update_reference(_normal_batch(500))
        for s in _drifted_batch(200):
            monitor._current.append(s)
        results = monitor.check_drift()
        drifted = [r for r in results if r.drifted]
        assert len(drifted) > 0

    def test_no_drift_on_same_distribution(self) -> None:
        """Identical distribution should not trigger drift."""
        rng = np.random.default_rng(42)
        monitor = DriftMonitor(current_window_size=200)
        monitor.update_reference(_normal_batch(500, rng=rng))
        for s in _normal_batch(200, rng=rng):
            monitor._current.append(s)
        results = monitor.check_drift()
        drifted = [r for r in results if r.drifted]
        assert len(drifted) == 0

    def test_check_drift_raises_if_reference_empty(self) -> None:
        """check_drift raises ValueError when reference window is empty."""
        monitor = DriftMonitor()
        monitor._current.append(_make_sample())
        with pytest.raises(ValueError, match="Reference window is empty"):
            monitor.check_drift()

    def test_drift_result_fields(self) -> None:
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
    def test_all_features_are_tested(self, feature) -> None:
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

    def test_get_monitor_returns_drift_monitor(self) -> None:
        """get_monitor() returns a DriftMonitor instance."""
        from app.monitoring import get_monitor

        monitor = get_monitor()
        assert isinstance(monitor, DriftMonitor)

    def test_get_monitor_returns_same_instance(self) -> None:
        """get_monitor() returns the same singleton on repeated calls."""
        from app.monitoring import get_monitor

        m1 = get_monitor()
        m2 = get_monitor()
        assert m1 is m2


class TestDriftMonitorReset:
    """Tests for DriftMonitor.reset() and drifted_features()."""

    def test_reset_clears_reference_window(self) -> None:
        """reset() empties the reference window."""
        monitor = DriftMonitor()
        monitor.update_reference(_normal_batch(100))
        assert monitor.reference_size == 100
        monitor.reset()
        assert monitor.reference_size == 0

    def test_reset_clears_current_window(self) -> None:
        """reset() empties the current window."""
        monitor = DriftMonitor(current_window_size=50)
        monitor.update_reference(_normal_batch(50))
        for s in _normal_batch(30):
            monitor._current.append(s)
        monitor.reset()
        assert monitor.current_size == 0

    def test_drifted_features_returns_correct_names(self) -> None:
        """drifted_features() filters DriftResult list to drifted names."""
        monitor = DriftMonitor(current_window_size=50)
        monitor.update_reference(_normal_batch(200))
        for s in _drifted_batch(50):
            monitor._current.append(s)
        results = monitor.check_drift()
        drifted = monitor.drifted_features(results)
        assert isinstance(drifted, list)
        assert all(name in FEATURE_COLS for name in drifted)

    def test_drifted_features_empty_when_no_drift(self) -> None:
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

    def test_repr_contains_feature_name(self) -> None:
        """DriftResult repr includes the feature name."""
        r = DriftResult(
            feature_name="cpu_usage_pct",
            ks_statistic=0.3,
            p_value=0.001,
            drifted=True,
        )
        assert "cpu_usage_pct" in repr(r)

    def test_repr_contains_drifted_status(self) -> None:
        """DriftResult repr says DRIFTED for a drifted result."""
        r = DriftResult(
            feature_name="latency_p99_ms",
            ks_statistic=0.45,
            p_value=0.0001,
            drifted=True,
        )
        assert "DRIFTED" in repr(r)

    def test_repr_contains_stable_status(self) -> None:
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

    def test_summary_returns_dict(self) -> None:
        """summary() returns a dict."""
        monitor = DriftMonitor()
        result = monitor.summary()
        assert isinstance(result, dict)

    def test_summary_has_reference_size(self) -> None:
        """summary() dict contains reference_size key."""
        monitor = DriftMonitor()
        assert "reference_size" in monitor.summary()

    def test_summary_has_current_size(self) -> None:
        """summary() dict contains current_size key."""
        monitor = DriftMonitor()
        assert "current_size" in monitor.summary()

    def test_summary_reference_size_matches(self) -> None:
        """summary reference_size matches actual reference window size."""
        monitor = DriftMonitor()
        monitor.update_reference(_normal_batch(50))
        assert monitor.summary()["reference_size"] == 50

    def test_summary_feature_count_correct(self) -> None:
        """summary feature_count equals FEATURE_COLS length."""
        monitor = DriftMonitor()
        assert monitor.summary()["feature_count"] == len(FEATURE_COLS)

    def test_summary_threshold_matches_init(self) -> None:
        """summary threshold matches the value passed at construction."""
        monitor = DriftMonitor(threshold=0.01)
        assert monitor.summary()["threshold"] == 0.01


class TestDriftMonitorStableAndRate:
    """Tests for stable_features() and drift_rate()."""

    def test_stable_features_complement_of_drifted(self) -> None:
        """stable_features() returns features not in drifted_features()."""
        monitor = DriftMonitor(current_window_size=50)
        rng = np.random.default_rng(42)
        monitor.update_reference(_normal_batch(200, rng=rng))
        for s in _normal_batch(50, rng=rng):
            monitor._current.append(s)
        results = monitor.check_drift()
        drifted = set(monitor.drifted_features(results))
        stable = set(monitor.stable_features(results))
        assert drifted | stable == set(FEATURE_COLS)
        assert drifted & stable == set()

    def test_drift_rate_between_zero_and_one(self) -> None:
        """drift_rate() is always in [0.0, 1.0]."""
        monitor = DriftMonitor(current_window_size=50)
        rng = np.random.default_rng(10)
        monitor.update_reference(_normal_batch(200, rng=rng))
        for s in _normal_batch(50, rng=rng):
            monitor._current.append(s)
        results = monitor.check_drift()
        rate = monitor.drift_rate(results)
        assert 0.0 <= rate <= 1.0

    def test_drift_rate_zero_for_empty_results(self) -> None:
        """drift_rate() returns 0.0 for an empty results list."""
        monitor = DriftMonitor()
        assert monitor.drift_rate([]) == 0.0

    def test_drift_rate_one_when_all_drifted(self) -> None:
        """drift_rate() returns 1.0 when all features drifted."""
        monitor = DriftMonitor()
        all_drifted = [DriftResult(f, 0.9, 0.0001, True) for f in FEATURE_COLS]
        assert monitor.drift_rate(all_drifted) == 1.0


class TestDriftResultToDict:
    """Tests for DriftResult.to_dict()."""

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() returns a dict."""
        r = DriftResult("cpu_usage_pct", 0.3, 0.001, True)
        assert isinstance(r.to_dict(), dict)

    def test_to_dict_has_all_keys(self) -> None:
        """to_dict() output has feature_name, ks_statistic, p_value, drifted, timestamp."""
        r = DriftResult("cpu_usage_pct", 0.3, 0.001, True)
        keys = {"feature_name", "ks_statistic", "p_value", "drifted", "timestamp"}
        assert keys.issubset(r.to_dict())

    def test_to_dict_drifted_matches(self) -> None:
        """to_dict() drifted key matches the DriftResult drifted field."""
        r = DriftResult("memory_usage_pct", 0.5, 0.0001, True)
        assert r.to_dict()["drifted"] is True

    def test_to_dict_timestamp_is_string(self) -> None:
        """to_dict() timestamp is an ISO-format string."""
        r = DriftResult("disk_io_util_pct", 0.1, 0.5, False)
        assert isinstance(r.to_dict()["timestamp"], str)

    @pytest.mark.parametrize("feature", FEATURE_COLS)
    def test_to_dict_feature_name_matches(self, feature: str) -> None:
        """to_dict() feature_name matches the DriftResult feature_name for all features."""
        r = DriftResult(feature, 0.2, 0.1, False)
        assert r.to_dict()["feature_name"] == feature


class TestDriftMonitorAnyDrifted:
    """Tests for DriftMonitor.any_drifted()."""

    def test_any_drifted_true_when_one_drifted(self) -> None:
        """any_drifted() returns True when at least one feature drifted."""
        monitor = DriftMonitor()
        results = [
            DriftResult("cpu_usage_pct", 0.9, 0.0001, True),
            DriftResult("memory_usage_pct", 0.05, 0.6, False),
        ]
        assert monitor.any_drifted(results) is True

    def test_any_drifted_false_when_none_drifted(self) -> None:
        """any_drifted() returns False when no features drifted."""
        monitor = DriftMonitor()
        results = [DriftResult(f, 0.05, 0.6, False) for f in FEATURE_COLS]
        assert monitor.any_drifted(results) is False

    def test_any_drifted_false_for_empty_list(self) -> None:
        """any_drifted() returns False for an empty results list."""
        assert DriftMonitor().any_drifted([]) is False

    @pytest.mark.parametrize(
        "drifted_flags,expected",
        [
            ([True, False, False], True),
            ([False, False, False], False),
            ([True, True, True], True),
        ],
    )
    def test_any_drifted_parametrized(self, drifted_flags, expected) -> None:
        """any_drifted returns expected value for mixed drift patterns."""
        monitor = DriftMonitor()
        results = [
            DriftResult(FEATURE_COLS[i], 0.3, 0.001, flag)
            for i, flag in enumerate(drifted_flags)
        ]
        assert monitor.any_drifted(results) is expected


class TestDriftMonitorRepr:
    """Tests for DriftMonitor.__repr__."""

    def test_repr_contains_threshold(self) -> None:
        """DriftMonitor repr includes the threshold value."""
        monitor = DriftMonitor(threshold=0.03)
        assert "0.03" in repr(monitor)

    def test_repr_contains_ref_size(self) -> None:
        """DriftMonitor repr includes reference window size."""
        monitor = DriftMonitor()
        monitor.update_reference(_normal_batch(50))
        assert "50" in repr(monitor)

    def test_repr_is_string(self) -> None:
        """DriftMonitor repr returns a string."""
        assert isinstance(repr(DriftMonitor()), str)

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.10])
    def test_repr_threshold_parametrized(self, threshold: float) -> None:
        """repr includes the configured threshold for any valid value."""
        monitor = DriftMonitor(threshold=threshold)
        assert str(threshold) in repr(monitor)


class TestDriftMonitorToDict:
    """Tests for DriftMonitor.to_dict()."""

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() returns a dict."""
        monitor = DriftMonitor()
        assert isinstance(monitor.to_dict(), dict)

    def test_to_dict_has_feature_cols(self) -> None:
        """to_dict() includes feature_cols key with all feature names."""
        monitor = DriftMonitor()
        d = monitor.to_dict()
        assert "feature_cols" in d
        assert set(d["feature_cols"]) == set(FEATURE_COLS)

    def test_to_dict_feature_count_correct(self) -> None:
        """to_dict() feature_count matches FEATURE_COLS length."""
        monitor = DriftMonitor()
        assert monitor.to_dict()["feature_count"] == len(FEATURE_COLS)

    def test_to_dict_threshold_matches(self) -> None:
        """to_dict() threshold matches the value at construction."""
        monitor = DriftMonitor(threshold=0.02)
        assert monitor.to_dict()["threshold"] == 0.02


class TestDriftMonitorParametrized:
    """Parametrized threshold and batch size tests for DriftMonitor."""

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.10, 0.20])
    def test_threshold_stored_correctly(self, threshold: float) -> None:
        """DriftMonitor stores the threshold passed at construction."""
        monitor = DriftMonitor(threshold=threshold)
        assert monitor.to_dict()["threshold"] == pytest.approx(threshold)

    @pytest.mark.parametrize("n_samples", [50, 100, 200, 500])
    def test_reference_size_after_update(self, n_samples: int) -> None:
        """reference_size equals exactly the number of samples in the reference window."""
        monitor = DriftMonitor()
        rng = np.random.default_rng(42)
        samples = [
            _make_sample(
                cpu=float(rng.uniform(20, 60)),
                mem=float(rng.uniform(30, 70)),
            )
            for _ in range(n_samples)
        ]
        monitor.update_reference(samples)
        assert monitor.reference_size == n_samples

    @pytest.mark.parametrize("feature", FEATURE_COLS)
    def test_feature_col_in_to_dict(self, feature: str) -> None:
        """Every expected feature column appears in to_dict output."""
        monitor = DriftMonitor()
        assert feature in monitor.to_dict()["feature_cols"]

    @pytest.mark.parametrize("n_ref,n_cur", [(100, 50), (200, 100), (500, 200)])
    def test_check_drift_returns_results_list(self, n_ref: int, n_cur: int) -> None:
        """check_drift() returns a list of DriftResult for different window sizes."""
        monitor = DriftMonitor(current_window_size=n_cur)
        rng = np.random.default_rng(7)
        monitor.update_reference(
            [_make_sample(cpu=float(rng.uniform(20, 60))) for _ in range(n_ref)]
        )
        for _ in range(n_cur):
            monitor._current.append(_make_sample(cpu=float(rng.uniform(20, 60))))
        results = monitor.check_drift()
        assert isinstance(results, list)
        assert all(isinstance(r, DriftResult) for r in results)


class TestDriftMonitorDriftSummary:
    """Tests for DriftMonitor.drift_summary()."""

    def _make_results(self, drifted_flags: list[bool]) -> list[DriftResult]:
        from app.monitoring import FEATURE_COLS
        from datetime import datetime
        results = []
        for i, flag in enumerate(drifted_flags):
            col = FEATURE_COLS[i % len(FEATURE_COLS)]
            results.append(
                DriftResult(
                    feature_name=col,
                    ks_statistic=0.3 if flag else 0.05,
                    p_value=0.01 if flag else 0.5,
                    drifted=flag,
                    timestamp=datetime.utcnow(),
                )
            )
        return results

    def test_summary_has_any_drifted_key(self) -> None:
        """drift_summary() result contains any_drifted key."""
        monitor = DriftMonitor()
        results = self._make_results([True, False])
        summary = monitor.drift_summary(results)
        assert "any_drifted" in summary

    def test_summary_has_drift_rate_key(self) -> None:
        """drift_summary() result contains drift_rate key."""
        monitor = DriftMonitor()
        results = self._make_results([True, False])
        summary = monitor.drift_summary(results)
        assert "drift_rate" in summary

    def test_summary_total_features_matches_results_length(self) -> None:
        """total_features equals the number of DriftResult objects passed."""
        monitor = DriftMonitor()
        results = self._make_results([True, False, True])
        summary = monitor.drift_summary(results)
        assert summary["total_features"] == 3

    @pytest.mark.parametrize("n_drifted,n_stable", [(0, 3), (1, 2), (2, 1), (3, 0)])
    def test_summary_drifted_plus_stable_equals_total(self, n_drifted: int, n_stable: int) -> None:
        """drifted_features + stable_features counts equal total_features."""
        monitor = DriftMonitor()
        flags = [True] * n_drifted + [False] * n_stable
        results = self._make_results(flags)
        summary = monitor.drift_summary(results)
        assert len(summary["drifted_features"]) + len(summary["stable_features"]) == summary["total_features"]
