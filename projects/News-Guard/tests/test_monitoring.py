"""Monitoring and drift detection tests for News-Guard."""


import pytest

from app.monitoring import compute_drift, get_monitoring_summary


def test_drift_detected_with_different_distributions():
    import numpy as np
    rng = np.random.default_rng(42)
    reference = list(rng.beta(2, 5, 200))
    current = list(rng.beta(5, 2, 200))
    result = compute_drift(reference, current)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.0


def test_no_drift_with_same_distribution():
    import numpy as np
    rng = np.random.default_rng(42)
    reference = list(rng.normal(0.5, 0.1, 200))
    current = list(rng.normal(0.5, 0.1, 200))
    result = compute_drift(reference, current)
    assert result["p_value"] >= 0.05


def test_drift_insufficient_samples():
    result = compute_drift([0.3, 0.4, 0.5], [0.6])
    assert result["drift_detected"] is False
    assert result["note"] == "insufficient_samples"


def test_drift_result_has_required_keys():
    result = compute_drift(list(range(50)), list(range(50, 100)))
    for key in ("ks_statistic", "p_value", "drift_detected"):
        assert key in result


def test_monitoring_summary_structure(db_session):
    summary = get_monitoring_summary(db_session)
    assert "total_predictions" in summary
    assert "fake_count" in summary
    assert "real_count" in summary
    assert "fake_rate" in summary
    assert isinstance(summary["drift_detected"], bool)


def test_monitoring_summary_fake_rate_bounds(db_session):
    summary = get_monitoring_summary(db_session)
    assert 0.0 <= summary["fake_rate"] <= 1.0


@pytest.mark.parametrize("ref,curr,expect_drift", [
    ([0.1] * 100, [0.9] * 100, True),
    ([0.5] * 100, [0.5] * 100, False),
])
def test_drift_parametrized(ref, curr, expect_drift):
    result = compute_drift(ref, curr)
    assert result["drift_detected"] == expect_drift
