"""End-to-end drift detection tests against a seeded database.

The unit tests in ``test_monitoring.py`` exercise ``compute_drift`` on plain
lists. These drive the full path -- seed both windows, query them back
through SQLAlchemy, and assert the API returns a correctly typed result --
which is where numpy scalars and empty-window handling actually surface.
"""

from __future__ import annotations

import json

import pytest

from app.monitoring import run_drift_check
from scripts.seed_data import seed


def test_drift_detected_on_shifted_window(db_session):
    """A tenfold volume shift must be flagged."""
    seed(db_session, n_reference=200, n_recent=60, drift=True)
    result = run_drift_check(db_session)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5
    assert result["p_value"] < 0.05


def test_no_drift_on_stable_traffic(db_session):
    """Two windows from the same distribution must not raise an alert."""
    seed(db_session, n_reference=200, n_recent=200, drift=False)
    result = run_drift_check(db_session)
    assert result["drift_detected"] is False


def test_drift_result_types_are_native_python(db_session):
    """scipy returns numpy scalars; the result must carry native floats."""
    seed(db_session, n_reference=100, n_recent=50, drift=True)
    result = run_drift_check(db_session)
    assert type(result["ks_statistic"]) is float
    assert type(result["p_value"]) is float
    assert type(result["drift_detected"]) is bool


def test_drift_result_is_json_serialisable(db_session):
    """The dict is returned straight from the endpoint, so it must serialise."""
    seed(db_session, n_reference=100, n_recent=50, drift=True)
    result = run_drift_check(db_session)
    assert json.loads(json.dumps(result)) == result


def test_drift_check_records_a_row(db_session):
    """Every check is persisted for later audit."""
    from app.database import DriftLog

    before = db_session.query(DriftLog).count()
    seed(db_session, n_reference=100, n_recent=50, drift=True)
    run_drift_check(db_session)
    assert db_session.query(DriftLog).count() == before + 1


def test_empty_database_reports_insufficient_data(db_session):
    """With no history the check must degrade, not raise."""
    result = run_drift_check(db_session)
    assert result["drift_detected"] is False
    assert "error" in result


@pytest.mark.parametrize("drift", [True, False])
def test_drift_endpoint_returns_200(client, db_session, drift):
    seed(db_session, n_reference=150, n_recent=60, drift=drift)
    resp = client.get("/api/v1/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"ks_statistic", "p_value", "drift_detected"}
    assert body["drift_detected"] is drift
