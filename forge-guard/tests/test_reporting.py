"""Tests for prediction and drift report export utilities."""

from __future__ import annotations

import csv
import io

import pytest

SENSOR = {
    "temperature": 75.0,
    "pressure": 50.0,
    "vibration": 2.0,
    "cycle_time": 30.0,
    "tool_wear": 10.0,
    "power_consumption": 100.0,
    "humidity": 45.0,
}


def _add_predictions(db_session, count: int = 3) -> None:
    from app.monitoring import log_prediction

    for i in range(count):
        log_prediction(
            db=db_session,
            sensor_data=SENSOR,
            prediction=i % 2,
            defect_probability=0.1 * (i + 1),
        )


def test_export_predictions_csv_empty_db(db_session):
    from app.reporting import export_predictions_csv

    result = export_predictions_csv(db_session)
    lines = result.strip().splitlines()
    assert len(lines) == 1  # header only


def test_export_predictions_csv_has_header(db_session):
    from app.reporting import export_predictions_csv

    result = export_predictions_csv(db_session)
    reader = csv.reader(io.StringIO(result))
    header = next(reader)
    assert "id" in header
    assert "prediction" in header
    assert "defect_probability" in header


def test_export_predictions_csv_includes_rows(db_session):
    from app.reporting import export_predictions_csv

    _add_predictions(db_session, 4)
    result = export_predictions_csv(db_session)
    lines = result.strip().splitlines()
    assert len(lines) == 5  # 1 header + 4 rows


def test_export_drift_reports_json_empty(db_session):
    from app.reporting import export_drift_reports_json

    result = export_drift_reports_json(db_session)
    assert result == []


def test_prediction_summary_json_empty(db_session):
    from app.reporting import prediction_summary_json

    result = prediction_summary_json(db_session)
    assert result["total"] == 0
    assert result["defect_rate"] == 0.0
    assert "exported_at" in result
    assert "defect_count" in result


def test_prediction_summary_json_with_data(db_session):
    from app.reporting import prediction_summary_json

    _add_predictions(db_session, 4)
    result = prediction_summary_json(db_session)
    assert result["total"] == 4
    assert 0.0 <= result["defect_rate"] <= 1.0


@pytest.mark.parametrize("hours", [1, 12, 24, 168])
def test_export_predictions_csv_window(db_session, hours):
    from app.reporting import export_predictions_csv

    _add_predictions(db_session, 2)
    result = export_predictions_csv(db_session, hours=hours)
    assert result  # should always have header


def test_export_predictions_csv_has_sensor_columns(db_session):
    from app.reporting import export_predictions_csv

    _add_predictions(db_session, 1)
    result = export_predictions_csv(db_session)
    reader = csv.reader(io.StringIO(result))
    header = next(reader)
    assert "temperature" in header
    assert "vibration" in header


def test_prediction_summary_json_keys(db_session):
    from app.reporting import prediction_summary_json

    result = prediction_summary_json(db_session)
    for key in ("total", "defect_count", "defect_rate", "exported_at"):
        assert key in result


def test_export_drift_with_data(db_session):
    from app.database import DriftReport
    from app.reporting import export_drift_reports_json

    db_session.add(
        DriftReport(
            feature="pressure",
            ks_statistic=0.35,
            p_value=0.02,
            drift_detected=True,
            window_size=300,
            model_version="1.0.0",
        )
    )
    db_session.commit()
    result = export_drift_reports_json(db_session)
    assert len(result) >= 1
    assert result[0]["feature"] == "pressure"


def test_export_predictions_csv_respects_limit(db_session):
    from app.reporting import export_predictions_csv

    _add_predictions(db_session, 5)
    result = export_predictions_csv(db_session, limit=2)
    lines = result.strip().splitlines()
    assert len(lines) <= 3  # header + up to 2 rows


@pytest.mark.parametrize("n_rows", [1, 3, 5, 10])
def test_export_predictions_csv_row_count(db_session, n_rows: int) -> None:
    from app.reporting import export_predictions_csv

    _add_predictions(db_session, n_rows)
    result = export_predictions_csv(db_session, limit=n_rows)
    lines = result.strip().splitlines()
    assert len(lines) == n_rows + 1  # header + n_rows


@pytest.mark.parametrize("n_preds", [2, 4, 6])
def test_prediction_summary_json_count(db_session, n_preds: int) -> None:
    from app.reporting import prediction_summary_json

    _add_predictions(db_session, n_preds)
    result = prediction_summary_json(db_session)
    assert result["total_predictions"] == n_preds


def test_csv_export_includes_field_names(db_session) -> None:
    from app.reporting import export_predictions_csv

    _add_predictions(db_session, 1)
    result = export_predictions_csv(db_session, limit=1)
    first_line = result.strip().splitlines()[0]
    assert "defect_probability" in first_line or "prediction" in first_line
