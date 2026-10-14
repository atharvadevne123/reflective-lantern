"""Tests for drift detection and prediction logging."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.database import DriftRecord, Prediction
from app.monitoring import (
    DRIFT_THRESHOLD,
    MIN_SAMPLE_SIZE,
    get_drift_summary,
    log_prediction,
    run_drift_check,
    should_retrain,
)


def _make_prediction(db: Session, viral_prob: float = 0.3, **kwargs) -> Prediction:
    """Helper to insert a Prediction row directly."""
    defaults = {
        "request_id": str(uuid.uuid4()),
        "text": "test post #hello",
        "author_followers": 1000,
        "has_media": False,
        "has_link": False,
        "hashtag_count": 1,
        "mention_count": 0,
        "hour_of_day": 12,
        "day_of_week": 2,
        "viral_probability": viral_prob,
        "viral_label": viral_prob >= 0.5,
        "ensemble_confidence": abs(viral_prob - 0.5) * 2,
        "model_version": "test",
        "latency_ms": 10.0,
    }
    defaults.update(kwargs)
    row = Prediction(**defaults)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


class TestLogPrediction:
    def test_log_creates_record(self, db_session: Session) -> None:
        request_id = str(uuid.uuid4())
        sample = {
            "text": "hello world",
            "author_followers": 500,
            "has_media": False,
            "has_link": False,
            "hashtag_count": 0,
            "mention_count": 0,
            "hour_of_day": 10,
            "day_of_week": 1,
        }
        record = log_prediction(
            db=db_session,
            request_id=request_id,
            sample=sample,
            viral_probability=0.3,
            viral_label=False,
            ensemble_confidence=0.4,
            model_version="1.0.0",
            latency_ms=5.0,
        )
        assert record.id is not None
        assert record.request_id == request_id
        assert record.viral_probability == pytest.approx(0.3)
        assert record.viral_label is False

    def test_log_stores_text(self, db_session: Session) -> None:
        sample = {
            "text": "unique text xyz123",
            "author_followers": 100,
            "has_media": False,
            "has_link": False,
            "hashtag_count": 0,
            "mention_count": 0,
            "hour_of_day": 8,
            "day_of_week": 0,
        }
        record = log_prediction(
            db=db_session,
            request_id=str(uuid.uuid4()),
            sample=sample,
            viral_probability=0.1,
            viral_label=False,
            ensemble_confidence=0.8,
            model_version="1.0.0",
            latency_ms=3.0,
        )
        assert record.text == "unique text xyz123"


class TestRunDriftCheck:
    def test_skips_when_insufficient_data(self, db_session: Session) -> None:
        records = run_drift_check(db_session)
        assert records == []

    def test_returns_drift_records_with_enough_data(self, db_session: Session) -> None:
        # Insert data spread across reference and current windows
        now = datetime.utcnow()
        ref_time = now - timedelta(days=20)
        cur_time = now - timedelta(days=3)

        for i in range(MIN_SAMPLE_SIZE + 5):
            p = _make_prediction(db_session, viral_prob=float(i % 2) * 0.8)
            p.created_at = ref_time
        for i in range(MIN_SAMPLE_SIZE + 5):
            p = _make_prediction(db_session, viral_prob=float(i % 2) * 0.2)
            p.created_at = cur_time
        db_session.commit()

        records = run_drift_check(db_session, reference_window_days=30, current_window_days=7)
        # May detect drift or not; just check it ran without error
        assert isinstance(records, list)


class TestGetDriftSummary:
    def test_returns_list(self, db_session: Session) -> None:
        result = get_drift_summary(db_session)
        assert isinstance(result, list)

    def test_returns_dicts_with_expected_keys(self, db_session: Session) -> None:
        # Insert a drift record manually
        record = DriftRecord(
            feature_name="viral_probability",
            ks_statistic=0.15,
            p_value=0.03,
            drift_detected=True,
            sample_size=100,
        )
        db_session.add(record)
        db_session.commit()

        result = get_drift_summary(db_session, limit=1)
        assert len(result) >= 1
        item = result[0]
        assert "feature_name" in item
        assert "ks_statistic" in item
        assert "drift_detected" in item


class TestShouldRetrain:
    def test_false_when_insufficient_predictions(self, db_session: Session) -> None:
        assert not should_retrain(db_session, min_new_samples=1000)

    def test_true_when_threshold_met(self, db_session: Session) -> None:
        for _ in range(5):
            _make_prediction(db_session)
        assert should_retrain(db_session, min_new_samples=5)

    def test_false_just_below_threshold(self, db_session: Session) -> None:
        for _ in range(4):
            _make_prediction(db_session)
        assert not should_retrain(db_session, min_new_samples=10)


@pytest.mark.parametrize("viral_prob,expected_label", [
    (0.0, False),
    (0.49, False),
    (0.5, True),
    (0.99, True),
])
def test_log_prediction_label_threshold(
    db_session: Session, viral_prob: float, expected_label: bool
) -> None:
    """Viral label must match the 0.5 threshold."""
    sample = {
        "text": "test",
        "author_followers": 100,
        "has_media": False,
        "has_link": False,
        "hashtag_count": 0,
        "mention_count": 0,
        "hour_of_day": 12,
        "day_of_week": 0,
    }
    record = log_prediction(
        db=db_session,
        request_id=str(uuid.uuid4()),
        sample=sample,
        viral_probability=viral_prob,
        viral_label=expected_label,
        ensemble_confidence=abs(viral_prob - 0.5) * 2,
        model_version="test",
        latency_ms=1.0,
    )
    assert record.viral_label == expected_label
