"""Extended anomaly detection tests for quake-net."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.anomaly import SeismicAnomalyDetector
from app.similarity import SIGNATURE_COLUMNS


def _make_events(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "depth_km": rng.uniform(0, 300, n),
            "p_wave_amplitude": rng.exponential(100, n),
            "s_wave_amplitude": rng.exponential(150, n),
            "epicentral_distance_km": rng.uniform(1, 500, n),
            "station_count": rng.integers(1, 50, n).astype(float),
        }
    )


def test_detector_not_fitted_on_init() -> None:
    det = SeismicAnomalyDetector()
    assert det.is_fitted is False


def test_score_before_fit_raises() -> None:
    det = SeismicAnomalyDetector()
    df = _make_events(10)
    with pytest.raises(ValueError, match="before fit"):
        det.score(df)


def test_fit_sets_fitted_flag() -> None:
    det = SeismicAnomalyDetector()
    det.fit(_make_events(100))
    assert det.is_fitted is True


def test_score_returns_list_of_dicts() -> None:
    det = SeismicAnomalyDetector()
    df = _make_events(50)
    det.fit(df)
    results = det.score(df)
    assert isinstance(results, list)
    assert len(results) == 50
    assert all(isinstance(r, dict) for r in results)


def test_score_result_has_required_keys() -> None:
    det = SeismicAnomalyDetector()
    df = _make_events(50)
    det.fit(df)
    results = det.score(df)
    for r in results[:5]:
        assert "is_anomaly" in r
        assert "anomaly_score" in r


def test_score_is_anomaly_is_bool() -> None:
    det = SeismicAnomalyDetector()
    df = _make_events(50)
    det.fit(df)
    results = det.score(df)
    for r in results:
        assert isinstance(r["is_anomaly"], bool)


def test_score_anomaly_score_is_finite() -> None:
    det = SeismicAnomalyDetector()
    df = _make_events(50)
    det.fit(df)
    results = det.score(df)
    for r in results:
        assert np.isfinite(r["anomaly_score"])


@pytest.mark.parametrize("contamination", [0.01, 0.05, 0.10, 0.20])
def test_contamination_affects_anomaly_rate(contamination: float) -> None:
    det = SeismicAnomalyDetector(contamination=contamination)
    df = _make_events(500)
    det.fit(df)
    results = det.score(df)
    anomaly_count = sum(1 for r in results if r["is_anomaly"])
    expected_min = int(500 * contamination * 0.5)
    assert anomaly_count >= expected_min


def test_fit_idempotent() -> None:
    det = SeismicAnomalyDetector()
    df = _make_events(100)
    det.fit(df)
    det.fit(df)
    assert det.is_fitted is True


def test_score_on_single_row_after_fit() -> None:
    det = SeismicAnomalyDetector()
    df = _make_events(100)
    det.fit(df)
    single = _make_events(1, seed=99)
    results = det.score(single)
    assert len(results) == 1


def test_outlier_cluster_flagged() -> None:
    rng = np.random.default_rng(0)
    normal = pd.DataFrame({col: rng.uniform(0, 10, 500) for col in SIGNATURE_COLUMNS})
    outliers = pd.DataFrame({col: np.full(5, 1e6) for col in SIGNATURE_COLUMNS})
    det = SeismicAnomalyDetector(contamination=0.05)
    det.fit(normal)
    results = det.score(outliers)
    anomaly_flags = [r["is_anomaly"] for r in results]
    assert sum(anomaly_flags) >= 3


@pytest.mark.parametrize("n_train", [50, 100, 200])
def test_fit_with_different_training_sizes(n_train: int) -> None:
    det = SeismicAnomalyDetector()
    df = _make_events(n_train)
    det.fit(df)
    results = det.score(_make_events(10))
    assert len(results) == 10
