"""Tests for Ops-Vision CRUD database operations."""


from app.crud import (
    avg_confidence,
    count_drift_alerts_last_24h,
    count_incidents_predicted,
    count_predictions,
    create_drift_alert,
    create_incident,
    create_prediction,
    get_incident,
    list_incidents,
)


class TestCreateIncident:
    """Tests for create_incident() CRUD function."""

    def _make_incident_data(self, service: str = "test-svc") -> dict:
        return {
            "service_name": service,
            "cpu_usage_pct": 85.0,
            "memory_usage_pct": 88.0,
            "error_rate_per_min": 60.0,
            "latency_p99_ms": 1500.0,
            "request_rate_per_sec": 45.0,
            "disk_io_util_pct": 80.0,
            "is_incident": True,
            "severity": "critical",
        }

    def test_create_returns_incident_with_id(self, db_session):
        """create_incident() returns an Incident with a populated id."""
        incident = create_incident(db_session, self._make_incident_data())
        assert incident.id is not None

    def test_get_incident_by_id(self, db_session):
        """get_incident() retrieves the record by primary key."""
        created = create_incident(db_session, self._make_incident_data())
        fetched = get_incident(db_session, created.id)
        assert fetched is not None
        assert fetched.service_name == "test-svc"

    def test_get_incident_returns_none_for_missing(self, db_session):
        """get_incident() returns None for a non-existent id."""
        result = get_incident(db_session, 999999)
        assert result is None

    def test_list_incidents_returns_all(self, db_session):
        """list_incidents() returns all created incidents."""
        create_incident(db_session, self._make_incident_data("svc-a"))
        create_incident(db_session, self._make_incident_data("svc-b"))
        results = list_incidents(db_session)
        assert len(results) >= 2

    def test_list_incidents_filters_by_service(self, db_session):
        """list_incidents() filters correctly by service_name."""
        create_incident(db_session, self._make_incident_data("target-svc"))
        create_incident(db_session, self._make_incident_data("other-svc"))
        results = list_incidents(db_session, service_name="target-svc")
        for r in results:
            assert r.service_name == "target-svc"


class TestCreatePrediction:
    """Tests for create_prediction() and aggregate functions."""

    def _make_prediction_data(self, is_incident: bool = True, confidence: float = 0.9) -> dict:
        return {
            "service_name": "pred-svc",
            "features": {"cpu_usage_pct": 85.0},
            "predicted_incident": is_incident,
            "predicted_severity": "critical" if is_incident else None,
            "confidence": confidence,
            "model_version": "1.0.0",
        }

    def test_create_prediction_returns_id(self, db_session):
        """create_prediction() returns a Prediction with populated id."""
        pred = create_prediction(db_session, self._make_prediction_data())
        assert pred.id is not None

    def test_count_predictions(self, db_session):
        """count_predictions() returns the correct count."""
        before = count_predictions(db_session)
        create_prediction(db_session, self._make_prediction_data())
        after = count_predictions(db_session)
        assert after == before + 1

    def test_count_incidents_predicted(self, db_session):
        """count_incidents_predicted() only counts incident=True predictions."""
        before = count_incidents_predicted(db_session)
        create_prediction(db_session, self._make_prediction_data(is_incident=True))
        create_prediction(db_session, self._make_prediction_data(is_incident=False))
        after = count_incidents_predicted(db_session)
        assert after == before + 1

    def test_avg_confidence(self, db_session):
        """avg_confidence() returns mean of confidence scores."""
        create_prediction(db_session, self._make_prediction_data(confidence=0.8))
        create_prediction(db_session, self._make_prediction_data(confidence=0.6))
        avg = avg_confidence(db_session)
        assert isinstance(avg, float)
        assert 0.0 <= avg <= 1.0


class TestCreateDriftAlert:
    """Tests for create_drift_alert() and related queries."""

    def _make_alert_data(self, drifted: bool = True) -> dict:
        return {
            "feature_name": "cpu_usage_pct",
            "ks_statistic": 0.35,
            "p_value": 0.001,
            "drifted": drifted,
            "reference_window": "2026-08-01",
            "current_window": "2026-08-25",
        }

    def test_create_drift_alert_returns_id(self, db_session):
        """create_drift_alert() returns a DriftAlert with id."""
        alert = create_drift_alert(db_session, self._make_alert_data())
        assert alert.id is not None

    def test_count_drift_alerts_last_24h(self, db_session):
        """count_drift_alerts_last_24h() counts recent drifted alerts."""
        before = count_drift_alerts_last_24h(db_session)
        create_drift_alert(db_session, self._make_alert_data(drifted=True))
        after = count_drift_alerts_last_24h(db_session)
        assert after == before + 1

    def test_non_drifted_alert_not_counted(self, db_session):
        """Non-drifted alerts are excluded from the 24h count."""
        before = count_drift_alerts_last_24h(db_session)
        create_drift_alert(db_session, self._make_alert_data(drifted=False))
        after = count_drift_alerts_last_24h(db_session)
        assert after == before


class TestGetPredictionByIdAndCountByService:
    """Tests for get_prediction_by_id() and count_predictions_by_service()."""

    def _make_prediction(self, service: str = "svc-a", confidence: float = 0.75) -> dict:
        return {
            "service_name": service,
            "features": {"cpu_usage_pct": 70.0},
            "predicted_incident": True,
            "predicted_severity": "high",
            "confidence": confidence,
            "model_version": "1.0.0",
        }

    def test_get_prediction_by_id_returns_record(self, db_session):
        """get_prediction_by_id() returns the created prediction."""
        from app.crud import create_prediction, get_prediction_by_id
        pred = create_prediction(db_session, self._make_prediction())
        fetched = get_prediction_by_id(db_session, pred.id)
        assert fetched is not None
        assert fetched.id == pred.id

    def test_get_prediction_by_id_none_for_missing(self, db_session):
        """get_prediction_by_id() returns None for unknown id."""
        from app.crud import get_prediction_by_id
        assert get_prediction_by_id(db_session, 9999999) is None

    def test_count_predictions_by_service(self, db_session):
        """count_predictions_by_service() counts only matching service rows."""
        from app.crud import count_predictions_by_service, create_prediction
        before = count_predictions_by_service(db_session, "svc-unique-x")
        create_prediction(db_session, self._make_prediction("svc-unique-x"))
        create_prediction(db_session, self._make_prediction("svc-unique-x"))
        create_prediction(db_session, self._make_prediction("other-svc"))
        after = count_predictions_by_service(db_session, "svc-unique-x")
        assert after == before + 2

    def test_count_predictions_by_service_zero_for_unknown(self, db_session):
        """count_predictions_by_service() returns 0 for an unknown service."""
        from app.crud import count_predictions_by_service
        assert count_predictions_by_service(db_session, "totally-unknown-svc-xyz") == 0
