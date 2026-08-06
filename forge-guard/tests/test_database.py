"""Tests for SQLAlchemy models and database session management."""

from __future__ import annotations

from datetime import datetime

from app.database import DriftReport, PredictionLog, RetrainingRun


def test_prediction_log_defaults(db_session):
    record = PredictionLog(
        temperature=75.0,
        pressure=50.0,
        vibration=2.0,
        cycle_time=30.0,
        tool_wear=10.0,
        power_consumption=100.0,
        humidity=45.0,
        prediction=0,
        defect_probability=0.1,
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    assert record.id is not None
    assert record.model_version == "1.0.0"
    assert isinstance(record.timestamp, datetime)


def test_drift_report_persists(db_session):
    report = DriftReport(
        feature="temperature",
        ks_statistic=0.42,
        p_value=0.001,
        drift_detected=True,
        window_size=500,
        model_version="1.0.0",
    )
    db_session.add(report)
    db_session.commit()
    db_session.refresh(report)
    assert report.id is not None
    assert report.drift_detected is True


def test_retraining_run_status_transitions(db_session):
    run = RetrainingRun(trigger="drift", auc_before=0.91, auc_after=0.94, status="success")
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    assert run.status == "success"
    assert run.auc_after > run.auc_before


def test_query_predictions_by_label(db_session):
    marker = "test-query-by-label"
    for label in (0, 0, 1):
        db_session.add(
            PredictionLog(
                temperature=75.0,
                pressure=50.0,
                vibration=2.0,
                cycle_time=30.0,
                tool_wear=10.0,
                power_consumption=100.0,
                humidity=45.0,
                prediction=label,
                defect_probability=float(label),
                correlation_id=marker,
            )
        )
    db_session.commit()
    defects = (
        db_session.query(PredictionLog)
        .filter(PredictionLog.prediction == 1, PredictionLog.correlation_id == marker)
        .count()
    )
    assert defects == 1


def test_prediction_log_correlation_id_nullable(db_session):
    record = PredictionLog(
        temperature=75.0, pressure=50.0, vibration=2.0, cycle_time=30.0,
        tool_wear=10.0, power_consumption=100.0, humidity=45.0,
        prediction=0, defect_probability=0.1,
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    assert record.correlation_id is None


def test_drift_report_no_drift(db_session):
    report = DriftReport(
        feature="humidity",
        ks_statistic=0.05,
        p_value=0.60,
        drift_detected=False,
        window_size=200,
        model_version="1.0.0",
    )
    db_session.add(report)
    db_session.commit()
    db_session.refresh(report)
    assert report.drift_detected is False
    assert report.p_value == 0.60


def test_retraining_run_defaults(db_session):
    run = RetrainingRun()
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    assert run.status == "running"
    assert run.trigger == "scheduled"
    assert run.finished_at is None


def test_multiple_prediction_logs_queried(db_session):
    for i in range(5):
        db_session.add(PredictionLog(
            temperature=float(70 + i), pressure=50.0, vibration=2.0,
            cycle_time=30.0, tool_wear=10.0, power_consumption=100.0, humidity=45.0,
            prediction=i % 2, defect_probability=float(i) / 10.0,
        ))
    db_session.commit()
    count = db_session.query(PredictionLog).count()
    assert count == 5


def test_prediction_log_model_version_default(db_session):
    record = PredictionLog(
        temperature=75.0, pressure=50.0, vibration=2.0, cycle_time=30.0,
        tool_wear=10.0, power_consumption=100.0, humidity=45.0,
        prediction=0, defect_probability=0.1,
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    assert record.model_version is not None


def test_drift_reports_bulk_insert(db_session):
    features = ["temperature", "pressure", "vibration"]
    for feat in features:
        db_session.add(DriftReport(
            feature=feat, ks_statistic=0.1, p_value=0.5,
            drift_detected=False, window_size=100, model_version="1.0.0",
        ))
    db_session.commit()
    count = db_session.query(DriftReport).count()
    assert count >= len(features)


def test_retraining_run_auc_values(db_session):
    run = RetrainingRun(auc_before=0.85, auc_after=0.89, status="success", trigger="api")
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    assert run.auc_before == 0.85
    assert run.auc_after == 0.89
    assert run.trigger == "api"


def test_prediction_log_all_fields_stored(db_session):
    record = PredictionLog(
        temperature=92.0, pressure=61.0, vibration=7.5,
        cycle_time=28.0, tool_wear=55.0, power_consumption=135.0, humidity=65.0,
        prediction=1, defect_probability=0.92, model_version="2.0.0",
        correlation_id="cid-full-test",
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    assert record.temperature == 92.0
    assert record.vibration == 7.5
    assert record.model_version == "2.0.0"
    assert record.correlation_id == "cid-full-test"
