"""Extended tests for ops-vision app/monitoring.py DriftMonitor."""

from __future__ import annotations

import pytest

from app.monitoring import DriftMonitor, DriftResult, get_monitor


FEATURES = ["cpu_util", "mem_util", "latency_ms"]


def _make_ref_samples(n: int = 50, value: float = 0.5) -> list[dict]:
    return [{f: value for f in FEATURES} for _ in range(n)]


class TestDriftMonitorReferenceSize:
    def test_zero_on_init(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=10)
        assert m.reference_size() == 0

    def test_grows_after_update(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=10)
        m.update_reference(_make_ref_samples(20))
        assert m.reference_size() == 20

    def test_reset_clears_reference(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=10)
        m.update_reference(_make_ref_samples(20))
        m.reset()
        assert m.reference_size() == 0


class TestDriftMonitorDriftDetection:
    def test_no_drift_on_similar_data(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=20)
        m.update_reference(_make_ref_samples(50, 0.5))
        for _ in range(30):
            m.record({f: 0.5 for f in FEATURES})
        results = m.check_drift()
        assert isinstance(results, list)

    def test_any_drifted_false_on_stable(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=20)
        m.update_reference(_make_ref_samples(50, 0.5))
        for _ in range(30):
            m.record({f: 0.5 for f in FEATURES})
        results = m.check_drift()
        assert isinstance(m.any_drifted(results), bool)

    def test_drift_summary_has_total_features(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=20)
        m.update_reference(_make_ref_samples(50))
        for _ in range(30):
            m.record({f: 0.5 for f in FEATURES})
        results = m.check_drift()
        summary = m.drift_summary(results)
        assert "total_features" in summary
        assert summary["total_features"] == len(FEATURES)


class TestDriftResult:
    def test_to_dict_has_feature(self) -> None:
        r = DriftResult(feature="cpu_util", ks_stat=0.1, p_value=0.5, drifted=False)
        d = r.to_dict()
        assert d["feature"] == "cpu_util"

    def test_repr_contains_feature_name(self) -> None:
        r = DriftResult(feature="latency_ms", ks_stat=0.3, p_value=0.02, drifted=True)
        assert "latency_ms" in repr(r)


class TestDriftMonitorDriftRate:
    def test_zero_on_no_results(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=10)
        assert m.drift_rate([]) == pytest.approx(0.0)

    def test_one_on_all_drifted(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=10)
        results = [
            DriftResult(feature=f, ks_stat=0.9, p_value=0.001, drifted=True)
            for f in FEATURES
        ]
        assert m.drift_rate(results) == pytest.approx(1.0)

    def test_half_drifted(self) -> None:
        m = DriftMonitor(features=FEATURES, min_reference=10)
        results = [
            DriftResult(feature=FEATURES[0], ks_stat=0.9, p_value=0.001, drifted=True),
            DriftResult(feature=FEATURES[1], ks_stat=0.1, p_value=0.5, drifted=False),
        ]
        assert m.drift_rate(results) == pytest.approx(0.5)
