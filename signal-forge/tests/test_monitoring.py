"""Drift detection and monitoring tests for Signal-Forge."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.monitoring import (
    DRIFT_THRESHOLD,
    FEATURE_NAMES,
    get_drift_summary,
    run_ks_drift_detection,
)


def test_no_drift_same_distribution(db_session):
    rng = np.random.default_rng(42)
    vals = rng.normal(0, 1, 100).tolist()
    result = run_ks_drift_detection(db_session, "volatility", vals, vals[:30])
    assert not result["drift_detected"]


def test_drift_detected_different_distribution(db_session):
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 100).tolist()
    cur = rng.normal(5, 1, 50).tolist()
    result = run_ks_drift_detection(db_session, "momentum", ref, cur)
    assert result["drift_detected"]
    assert result["p_value"] < DRIFT_THRESHOLD


def test_ks_returns_correct_keys(db_session):
    vals = list(range(50))
    result = run_ks_drift_detection(db_session, "beta", vals, vals[:20])
    expected = {"feature", "ks_statistic", "p_value", "drift_detected", "reference_mean", "current_mean"}
    assert expected.issubset(result.keys())


def test_insufficient_data_returns_early(db_session):
    result = run_ks_drift_detection(db_session, "volume_ratio", [1, 2, 3], [1])
    assert result.get("drift_detected") is False


def test_drift_summary_returns_dict(db_session):
    summary = get_drift_summary(db_session)
    assert "total_checks" in summary
    assert "drift_events" in summary
    assert isinstance(summary["total_checks"], int)


def test_feature_names_constant():
    assert len(FEATURE_NAMES) == 5
    assert "volatility" in FEATURE_NAMES
    assert "beta" in FEATURE_NAMES


@pytest.mark.parametrize("feature", FEATURE_NAMES)
def test_ks_drift_each_feature(db_session, feature):
    rng = np.random.default_rng(99)
    ref = rng.normal(0, 1, 80).tolist()
    cur = rng.normal(0, 1, 30).tolist()
    result = run_ks_drift_detection(db_session, feature, ref, cur)
    assert "ks_statistic" in result
    assert 0.0 <= result["ks_statistic"] <= 1.0
