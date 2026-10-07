"""Anomaly detection tests for Sports-Oracle."""

from __future__ import annotations

import pytest


def test_zscore_detects_outlier():
    from app.anomaly import zscore_anomalies
    import numpy as np

    values = [1.0] * 50 + [100.0]  # clear outlier at end
    anomalies = zscore_anomalies(values)
    assert 50 in anomalies


def test_zscore_clean_data():
    from app.anomaly import zscore_anomalies
    import numpy as np

    values = [float(i) for i in range(50)]
    anomalies = zscore_anomalies(values, threshold=5.0)
    assert len(anomalies) == 0


def test_iqr_detects_outlier():
    from app.anomaly import iqr_anomalies

    values = [0.5] * 100 + [50.0]
    anomalies = iqr_anomalies(values)
    assert 100 in anomalies


def test_detect_anomalies_returns_report():
    from app.anomaly import detect_anomalies

    values = [0.5] * 100 + [99.9]
    report = detect_anomalies(values)
    assert "n_anomalies" in report
    assert report["n_samples"] == 101
    assert report["anomaly_rate"] >= 0.0


def test_detect_anomalies_empty():
    from app.anomaly import detect_anomalies

    report = detect_anomalies([])
    assert report["n_anomalies"] == 0
