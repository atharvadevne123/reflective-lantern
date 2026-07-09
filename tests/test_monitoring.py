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
