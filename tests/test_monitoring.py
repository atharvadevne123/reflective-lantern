"""Tests for drift detection and prediction logging."""

import pytest

from app.monitoring import compute_drift, log_prediction, run_drift_check


def test_compute_drift_no_drift() -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    ref = rng.normal(500_000, 50_000, 100).tolist()
    cur = rng.normal(500_000, 50_000, 100).tolist()
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert isinstance(result["drift_detected"], bool)
    assert result["p_value"] > 0.05


def test_compute_drift_detects_shift() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    ref = rng.normal(300_000, 20_000, 200).tolist()
    cur = rng.normal(700_000, 20_000, 200).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


@pytest.mark.parametrize(
    "ref_mean,cur_mean,expect_drift",
    [
        (400_000, 400_000, False),
        (400_000, 900_000, True),
    ],
)
def test_compute_drift_parametrized(ref_mean, cur_mean, expect_drift) -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    ref = rng.normal(ref_mean, 10_000, 200).tolist()
    cur = rng.normal(cur_mean, 10_000, 200).tolist()
    result = compute_drift(ref, cur)
    assert result["drift_detected"] == expect_drift


def test_compute_drift_sample_counts() -> None:
    ref = [1.0] * 50
    cur = [2.0] * 30
    result = compute_drift(ref, cur)
    assert result["n_reference"] == 50
    assert result["n_current"] == 30


def test_log_prediction_creates_record(db_session) -> None:
    record = log_prediction(
        db=db_session,
        predicted_value=450_000.0,
        features={"sqft": 1800, "bedrooms": 3},
        correlation_id="test-corr-001",
        investment_score=6.5,
    )
    assert record.id is not None
    assert record.predicted_value == 450_000.0
    assert record.correlation_id == "test-corr-001"
    assert record.investment_score == 6.5


def test_run_drift_check_creates_report(db_session) -> None:
    import numpy as np

    rng = np.random.default_rng(5)
    ref = rng.normal(500_000, 30_000, 100).tolist()
    cur = rng.normal(800_000, 30_000, 100).tolist()
    report = run_drift_check(db_session, "test_feature", ref, cur)
    assert report.id is not None
    assert report.feature_name == "test_feature"
    assert report.drift_detected is True


def test_log_prediction_without_investment_score(db_session) -> None:
    record = log_prediction(
        db=db_session,
        predicted_value=300_000.0,
        features={"sqft": 1000},
        correlation_id="test-corr-002",
    )
    assert record.investment_score is None


def test_compute_drift_ks_statistic_range() -> None:
    import numpy as np

    rng = np.random.default_rng(99)
    ref = rng.uniform(0, 1, 100).tolist()
    cur = rng.uniform(0, 1, 100).tolist()
    result = compute_drift(ref, cur)
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_compute_drift_p_value_range() -> None:
    import numpy as np

    rng = np.random.default_rng(77)
    ref = rng.normal(0, 1, 200).tolist()
    cur = rng.normal(0, 1, 200).tolist()
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
