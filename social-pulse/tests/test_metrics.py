"""Tests for the prediction metrics module."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from app.database import DriftRecord, Prediction
from app.metrics import PredictionMetrics, compute_metrics


def _insert_prediction(db: Session, viral: bool = False, latency: float = 10.0) -> Prediction:
    p = Prediction(
        request_id=str(uuid.uuid4()),
        text="test",
        author_followers=1000,
        has_media=False,
        has_link=False,
        hashtag_count=0,
        mention_count=0,
        hour_of_day=12,
        day_of_week=2,
        viral_probability=0.9 if viral else 0.1,
        viral_label=viral,
        ensemble_confidence=0.8,
        model_version="1.0.0",
        latency_ms=latency,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


class TestComputeMetrics:
    def test_zero_predictions(self, db_session: Session) -> None:
        m = compute_metrics(db_session, "1.0.0")
        assert m.total_predictions == 0
        assert m.viral_rate == 0.0

    def test_counts_predictions(self, db_session: Session) -> None:
        _insert_prediction(db_session, viral=False)
        _insert_prediction(db_session, viral=True)
        m = compute_metrics(db_session, "1.0.0")
        assert m.total_predictions == 2

    def test_viral_rate(self, db_session: Session) -> None:
        for _ in range(3):
            _insert_prediction(db_session, viral=False)
        _insert_prediction(db_session, viral=True)
        m = compute_metrics(db_session, "1.0.0")
        assert abs(m.viral_rate - 0.25) < 0.01

    def test_avg_latency(self, db_session: Session) -> None:
        _insert_prediction(db_session, latency=10.0)
        _insert_prediction(db_session, latency=20.0)
        m = compute_metrics(db_session, "1.0.0")
        assert abs(m.avg_latency_ms - 15.0) < 0.1

    def test_drifted_features_empty_initially(self, db_session: Session) -> None:
        m = compute_metrics(db_session, "1.0.0")
        assert m.drifted_features == []

    def test_returns_metrics_instance(self, db_session: Session) -> None:
        m = compute_metrics(db_session, "1.0.0")
        assert isinstance(m, PredictionMetrics)


class TestPredictionMetricsToDict:
    def test_to_dict_all_keys(self, db_session: Session) -> None:
        m = compute_metrics(db_session, "1.0.0")
        d = m.to_dict()
        assert "total_predictions" in d
        assert "viral_rate" in d
        assert "avg_confidence" in d
        assert "avg_latency_ms" in d
        assert "model_version" in d
        assert "drifted_features" in d

    def test_viral_rate_rounded(self, db_session: Session) -> None:
        _insert_prediction(db_session, viral=True)
        _insert_prediction(db_session, viral=False)
        _insert_prediction(db_session, viral=False)
        m = compute_metrics(db_session, "1.0.0")
        d = m.to_dict()
        # Should be rounded to 4 decimal places
        assert len(str(d["viral_rate"]).split(".")[-1]) <= 4


@pytest.mark.parametrize("n_viral,n_total,expected_rate", [
    (0, 10, 0.0),
    (5, 10, 0.5),
    (10, 10, 1.0),
])
def test_viral_rate_parametrized(
    db_session: Session, n_viral: int, n_total: int, expected_rate: float
) -> None:
    for i in range(n_total):
        _insert_prediction(db_session, viral=(i < n_viral))
    m = compute_metrics(db_session, "1.0.0")
    assert abs(m.viral_rate - expected_rate) < 0.01
