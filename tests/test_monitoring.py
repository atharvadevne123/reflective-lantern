"""Drift detection and monitoring tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    LatencyTimer,
    compute_drift,
    set_reference_window,
)


def test_compute_drift_no_drift():
    ref = list(np.random.default_rng(1).normal(10, 2, 200))
    cur = list(np.random.default_rng(2).normal(10, 2, 200))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert not result["drift_detected"]


def test_compute_drift_detects_shift():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    cur = list(np.random.default_rng(2).normal(30, 1, 200))
    result = compute_drift(ref, cur)
    assert result["drift_detected"], f"Expected drift, p={result['p_value']}"
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data():
    result = compute_drift([1.0, 2.0], [3.0, 4.0])
    assert not result["drift_detected"]
    assert result["reason"] == "insufficient_data"


def test_compute_drift_p_value_range():
    ref = list(range(100))
    cur = list(range(100, 200))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["p_value"] <= 1.0
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_set_reference_window():
    values = list(range(600))
    set_reference_window(values)
    from app.monitoring import _reference_window

    assert len(_reference_window) == 500


def test_latency_timer():
    import time

    with LatencyTimer() as t:
        time.sleep(0.01)
    assert t.ms >= 5.0


def test_log_prediction(db_session):
    from datetime import datetime

    from app.monitoring import log_prediction

    log_prediction(db_session, "bldg-test", datetime.utcnow(), 15.5, 12.3)
    from app.database import PredictionLog

    count = db_session.query(PredictionLog).filter(PredictionLog.building_id == "bldg-test").count()
    assert count >= 1


def test_log_anomaly(db_session):
    from datetime import datetime

    from app.monitoring import log_anomaly

    log_anomaly(db_session, "bldg-test", datetime.utcnow(), 99.9, -0.6, 1, "critical")
    from app.database import AnomalyLog

    count = db_session.query(AnomalyLog).filter(AnomalyLog.building_id == "bldg-test").count()
    assert count >= 1


@pytest.mark.parametrize("mean_shift", [0, 5, 10, 20])
def test_drift_various_shifts(mean_shift: float):
    rng = np.random.default_rng(42)
    ref = list(rng.normal(10, 2, 200))
    cur = list(rng.normal(10 + mean_shift, 2, 200))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["p_value"] <= 1.0


def test_log_prediction_model_version_stored(db_session) -> None:
    record = log_prediction(
        db=db_session,
        predicted_value=500_000.0,
        features={"sqft": 2000},
        correlation_id="test-mv-001",
        model_version="2.0.0",
    )
    assert record.model_version == "2.0.0"


@pytest.mark.parametrize("predicted_value", [100_000.0, 500_000.0, 2_000_000.0])
def test_log_prediction_various_values(db_session, predicted_value) -> None:
    record = log_prediction(
        db=db_session,
        predicted_value=predicted_value,
        features={},
        correlation_id="test-pv",
    )
    assert record.predicted_value == pytest.approx(predicted_value)


def test_compute_drift_identical_distributions_no_drift() -> None:
    import numpy as np

    rng = np.random.default_rng(123)
    data = rng.normal(500_000, 30_000, 100).tolist()
    result = compute_drift(data, data)
    assert result["drift_detected"] is False
    assert result["ks_statistic"] == pytest.approx(0.0)


@pytest.mark.parametrize("n_samples", [20, 50, 200])
def test_compute_drift_various_sample_sizes(n_samples) -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, n_samples).tolist()
    cur = rng.normal(0, 1, n_samples).tolist()
    result = compute_drift(ref, cur)
    assert result["n_reference"] == n_samples
    assert result["n_current"] == n_samples


def test_run_drift_check_no_drift_report(db_session) -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    data = rng.normal(500_000, 30_000, 100).tolist()
    report = run_drift_check(db_session, "stable_feature", data, data)
    assert report.feature_name == "stable_feature"
    assert report.drift_detected is False


def test_log_prediction_features_stored_as_json(db_session) -> None:
    import json

    features = {"sqft": 2500, "bedrooms": 4, "bathrooms": 3.0}
    record = log_prediction(
        db=db_session,
        predicted_value=750_000.0,
        features=features,
        correlation_id="test-json",
    )
    assert record.id is not None
    stored = json.loads(record.features_json) if record.features_json else {}
    assert stored.get("sqft") == 2500


def test_min_drift_samples_constant() -> None:
    from app.monitoring import MIN_DRIFT_SAMPLES

    assert MIN_DRIFT_SAMPLES >= 2


def test_compute_drift_raises_on_too_few_samples() -> None:
    import pytest

    from app.monitoring import compute_drift

    with pytest.raises(ValueError, match="at least"):
        compute_drift([1.0], [2.0, 3.0])


def test_compute_drift_raises_when_current_too_small() -> None:
    import pytest

    from app.monitoring import compute_drift

    with pytest.raises(ValueError):
        compute_drift([1.0, 2.0], [3.0])


def test_drift_threshold_constant() -> None:
    from app.monitoring import DRIFT_THRESHOLD

    assert 0.0 < DRIFT_THRESHOLD < 1.0


def test_reference_window_constant() -> None:
    from app.monitoring import REFERENCE_WINDOW

    assert REFERENCE_WINDOW > 0


def test_default_model_version_is_string() -> None:
    from app.monitoring import DEFAULT_MODEL_VERSION

    assert isinstance(DEFAULT_MODEL_VERSION, str)
    assert len(DEFAULT_MODEL_VERSION) > 0
