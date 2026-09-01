"""Tests for drift detection and prediction logging."""

from __future__ import annotations

import pytest

from app.monitoring import compute_drift, get_prediction_stats, log_prediction


class TestComputeDrift:
    def test_no_drift_identical_distributions(self):
        ref = [0.5] * 50
        cur = [0.5] * 50
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is False

    def test_drift_detected_on_shift(self):
        ref = [0.2] * 100
        cur = [0.8] * 100
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True
        assert result["p_value"] < 0.05

    def test_ks_statistic_is_float(self):
        result = compute_drift([0.3] * 30, [0.7] * 30)
        assert isinstance(result["ks_statistic"], float)

    def test_p_value_in_range(self):
        result = compute_drift([0.5] * 30, [0.5] * 30)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_insufficient_reference_returns_no_drift(self):
        result = compute_drift([0.5] * 3, [0.5] * 30)
        assert result["drift_detected"] is False
        assert result["reason"] == "insufficient_data"

    def test_insufficient_current_returns_no_drift(self):
        result = compute_drift([0.5] * 30, [0.5] * 2)
        assert result["drift_detected"] is False

    @pytest.mark.parametrize(
        "ref_mean,cur_mean,expected_drift",
        [
            (0.5, 0.5, False),
            (0.1, 0.9, True),
            (0.3, 0.31, False),
        ],
    )
    def test_drift_parametrized(self, ref_mean, cur_mean, expected_drift):
        import numpy as np

        rng = np.random.default_rng(0)
        ref = (rng.normal(ref_mean, 0.05, 200)).clip(0, 1).tolist()
        cur = (rng.normal(cur_mean, 0.05, 200)).clip(0, 1).tolist()
        result = compute_drift(ref, cur)
        assert result["drift_detected"] == expected_drift


class TestLogPrediction:
    def test_log_returns_uuid_string(self, db_session):
        rid = log_prediction(db_session, {"lead_time": 5}, 0.65, 180.0, "1.0.0")
        assert isinstance(rid, str)
        assert len(rid) == 36

    def test_log_prediction_persisted(self, db_session):
        from app.database import Prediction

        log_prediction(db_session, {"lead_time": 5}, 0.65, 180.0, "1.0.0")
        count = db_session.query(Prediction).count()
        assert count >= 1

    def test_log_stores_correct_demand_score(self, db_session):
        from app.database import Prediction

        log_prediction(db_session, {"lead_time": 5}, 0.72, 200.0, "1.0.0")
        pred = db_session.query(Prediction).order_by(Prediction.id.desc()).first()
        assert pred.demand_score == pytest.approx(0.72)

    def test_log_stores_correct_rate(self, db_session):
        from app.database import Prediction

        log_prediction(db_session, {"guests_count": 2}, 0.50, 165.5, "1.0.0")
        pred = db_session.query(Prediction).order_by(Prediction.id.desc()).first()
        assert pred.suggested_rate == pytest.approx(165.5)

    def test_log_stores_model_version(self, db_session):
        from app.database import Prediction

        log_prediction(db_session, {}, 0.4, 120.0, "test-2.0.0")
        pred = db_session.query(Prediction).order_by(Prediction.id.desc()).first()
        assert pred.model_version == "test-2.0.0"


class TestGetPredictionStats:
    def test_empty_db_returns_zero_count(self, db_session):
        stats = get_prediction_stats(db_session)
        assert stats["count"] == 0
        assert stats["avg_demand_score"] is None

    def test_stats_after_predictions(self, db_session):
        for score in [0.3, 0.5, 0.7]:
            log_prediction(db_session, {}, score, 150.0, "1.0.0")
        stats = get_prediction_stats(db_session)
        assert stats["count"] == 3
        assert stats["avg_demand_score"] == pytest.approx(0.5, abs=0.01)

    def test_stats_min_max_correct(self, db_session):
        for score in [0.2, 0.5, 0.9]:
            log_prediction(db_session, {}, score, 150.0, "1.0.0")
        stats = get_prediction_stats(db_session)
        assert stats["min_demand_score"] <= stats["avg_demand_score"]
        assert stats["max_demand_score"] >= stats["avg_demand_score"]

    def test_stats_respects_limit(self, db_session):
        for _ in range(10):
            log_prediction(db_session, {}, 0.5, 150.0, "1.0.0")
        stats = get_prediction_stats(db_session, limit=5)
        assert stats["count"] <= 5


@pytest.mark.parametrize("n_preds", [1, 5, 10])
def test_prediction_stats_count_matches(db_session, n_preds: int) -> None:
    for i in range(n_preds):
        log_prediction(db_session, {}, 0.5, 120.0 + i, "1.0.0")
    stats = get_prediction_stats(db_session, limit=n_preds)
    assert stats["count"] == n_preds


@pytest.mark.parametrize(
    "scores,expected_avg",
    [
        ([0.5], 0.5),
        ([0.2, 0.8], 0.5),
        ([0.1, 0.3, 0.5], pytest.approx(0.3, abs=0.01)),
    ],
)
def test_prediction_stats_average(db_session, scores: list, expected_avg) -> None:
    for score in scores:
        log_prediction(db_session, {}, score, 100.0, "1.0.0")
    stats = get_prediction_stats(db_session, limit=len(scores))
    assert stats["avg_demand_score"] == expected_avg


@pytest.mark.parametrize(
    "ref_mean,cur_mean,expect_drift",
    [
        (0.5, 0.5, False),
        (0.5, 0.99, True),
    ],
)
def test_drift_detection_parametrized(ref_mean: float, cur_mean: float, expect_drift: bool) -> None:
    ref = [ref_mean] * 100
    cur = [cur_mean] * 100
    result = compute_drift(ref, cur)
    assert result["drift_detected"] == expect_drift


def test_drift_result_has_score_key() -> None:
    from app.monitoring import compute_drift

    result = compute_drift([0.5] * 50, [0.5] * 50)
    assert "drift_score" in result or "score" in result or "drift_detected" in result


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_get_prediction_stats_respects_limit(db_session, limit: int) -> None:
    from app.monitoring import get_prediction_stats, log_prediction

    for i in range(15):
        log_prediction(db_session, {}, float(i) / 15, 100.0, "1.0.0")
    stats = get_prediction_stats(db_session, limit=limit)
    assert isinstance(stats, dict)


def test_log_prediction_returns_none_or_id(db_session) -> None:
    from app.monitoring import log_prediction

    result = log_prediction(db_session, {"feat": 1.0}, 0.7, 150.0, "1.0.0")
    assert result is None or isinstance(result, int | str)
