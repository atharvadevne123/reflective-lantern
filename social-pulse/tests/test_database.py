"""Tests for SQLAlchemy database models and session management."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from sqlalchemy.orm import Session

from app.database import DriftRecord, Prediction, TrainingRun


class TestPredictionModel:
    def test_create_prediction(self, db_session: Session) -> None:
        rid = str(uuid.uuid4())
        pred = Prediction(
            request_id=rid,
            text="test post",
            author_followers=1000,
            has_media=False,
            has_link=False,
            hashtag_count=1,
            mention_count=0,
            hour_of_day=12,
            day_of_week=2,
            viral_probability=0.4,
            viral_label=False,
            ensemble_confidence=0.8,
            model_version="1.0.0",
            latency_ms=5.0,
        )
        db_session.add(pred)
        db_session.commit()
        db_session.refresh(pred)
        assert pred.id is not None
        assert pred.request_id == rid

    def test_unique_request_id(self, db_session: Session) -> None:
        rid = str(uuid.uuid4())
        p1 = Prediction(
            request_id=rid, text="a", author_followers=0,
            has_media=False, has_link=False, hashtag_count=0,
            mention_count=0, hour_of_day=0, day_of_week=0,
            viral_probability=0.1, viral_label=False,
            ensemble_confidence=0.8, model_version="1.0", latency_ms=1.0,
        )
        p2 = Prediction(
            request_id=rid, text="b", author_followers=0,
            has_media=False, has_link=False, hashtag_count=0,
            mention_count=0, hour_of_day=0, day_of_week=0,
            viral_probability=0.2, viral_label=False,
            ensemble_confidence=0.6, model_version="1.0", latency_ms=1.0,
        )
        db_session.add(p1)
        db_session.commit()
        db_session.add(p2)
        with pytest.raises(Exception):
            db_session.commit()

    def test_created_at_default(self, db_session: Session) -> None:
        pred = Prediction(
            request_id=str(uuid.uuid4()), text="t", author_followers=0,
            has_media=False, has_link=False, hashtag_count=0,
            mention_count=0, hour_of_day=0, day_of_week=0,
            viral_probability=0.5, viral_label=True,
            ensemble_confidence=0.0, model_version="1.0", latency_ms=1.0,
        )
        db_session.add(pred)
        db_session.commit()
        db_session.refresh(pred)
        assert pred.created_at is not None


class TestDriftRecordModel:
    def test_create_drift_record(self, db_session: Session) -> None:
        rec = DriftRecord(
            feature_name="viral_probability",
            ks_statistic=0.12,
            p_value=0.03,
            drift_detected=True,
            sample_size=100,
        )
        db_session.add(rec)
        db_session.commit()
        db_session.refresh(rec)
        assert rec.id is not None
        assert rec.drift_detected is True


class TestTrainingRunModel:
    def test_create_training_run(self, db_session: Session) -> None:
        run = TrainingRun(
            run_id=str(uuid.uuid4())[:8],
            model_version="1.0.0",
            auc_roc=0.82,
            f1_score=0.71,
            train_samples=500,
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        assert run.id is not None
        assert run.status == "running"

    def test_default_status(self, db_session: Session) -> None:
        run = TrainingRun(
            run_id="test-run-01",
            model_version="1.0.0",
            train_samples=100,
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        assert run.status == "running"


@pytest.mark.parametrize("viral_prob", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_prediction_viral_probability_values(db_session: Session, viral_prob: float) -> None:
    """Predictions should persist for any valid probability value."""
    pred = Prediction(
        request_id=str(uuid.uuid4()), text="test",
        author_followers=100, has_media=False, has_link=False,
        hashtag_count=0, mention_count=0, hour_of_day=10, day_of_week=1,
        viral_probability=viral_prob, viral_label=viral_prob >= 0.5,
        ensemble_confidence=abs(viral_prob - 0.5) * 2,
        model_version="1.0", latency_ms=2.0,
    )
    db_session.add(pred)
    db_session.commit()
    db_session.refresh(pred)
    assert abs(pred.viral_probability - viral_prob) < 1e-6
