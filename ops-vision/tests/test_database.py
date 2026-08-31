"""Tests for Ops-Vision database utilities."""


from app.database import Base, DriftAlert, Incident, Prediction, ping_db


class TestDatabaseModels:
    """Tests for ORM model table structure."""

    def test_incident_has_id_column(self):
        """Incident model has an 'id' primary key column."""
        cols = [c.name for c in Incident.__table__.columns]
        assert "id" in cols

    def test_incident_has_service_name(self):
        """Incident model has a 'service_name' column."""
        cols = [c.name for c in Incident.__table__.columns]
        assert "service_name" in cols

    def test_prediction_has_confidence(self):
        """Prediction model has a 'confidence' column."""
        cols = [c.name for c in Prediction.__table__.columns]
        assert "confidence" in cols

    def test_prediction_has_model_version(self):
        """Prediction model has a 'model_version' column."""
        cols = [c.name for c in Prediction.__table__.columns]
        assert "model_version" in cols

    def test_drift_alert_has_ks_statistic(self):
        """DriftAlert model has a 'ks_statistic' column."""
        cols = [c.name for c in DriftAlert.__table__.columns]
        assert "ks_statistic" in cols

    def test_drift_alert_has_drifted(self):
        """DriftAlert model has a 'drifted' boolean column."""
        cols = [c.name for c in DriftAlert.__table__.columns]
        assert "drifted" in cols

    def test_all_models_registered_in_base(self):
        """All three ORM models are registered with the shared Base."""
        table_names = set(Base.metadata.tables.keys())
        assert {"incidents", "predictions", "drift_alerts"}.issubset(table_names)


class TestPingDb:
    """Tests for ping_db() connectivity check."""

    def test_ping_db_returns_bool(self, db_session):
        """ping_db() returns a boolean."""
        result = ping_db()
        assert isinstance(result, bool)
