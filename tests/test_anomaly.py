"""Extended anomaly analysis tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.anomaly import compute_severity, iqr_flag, zscore_flag


def test_zscore_normal():
    assert not zscore_flag(10.0, mean=10.0, std=2.0, threshold=3.0)


def test_zscore_outlier():
    assert zscore_flag(20.0, mean=10.0, std=1.0, threshold=3.0)


def test_zscore_zero_std():
    assert not zscore_flag(5.0, mean=5.0, std=0.0)

    def test_insufficient_reference(self):
        result = quick_anomaly_check([3000.0] * 5, [3000.0, 9000.0])
        assert "error" in result

def test_iqr_within_fence():
    assert not iqr_flag(10.0, q1=8.0, q3=12.0)


def test_iqr_outside_fence():
    assert iqr_flag(30.0, q1=8.0, q3=12.0)


def test_iqr_below_fence():
    assert iqr_flag(-5.0, q1=8.0, q3=12.0)


def test_compute_severity_none():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(10.0, ref)
    assert result["severity"] == "none"


def test_compute_severity_critical():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(100.0, ref)
    assert result["severity"] == "critical"
    assert result["z_flag"] is True
    assert result["iqr_flag"] is True


@pytest.mark.parametrize("value,expected", [(10.0, "none"), (100.0, "critical")])
def test_compute_severity_parametrized(value, expected):
    ref = list(np.random.default_rng(42).normal(10, 1, 100))
    result = compute_severity(value, ref)
    assert result["severity"] == expected
