"""Database model tests for Risk-Sense."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, CreditApplication, DriftLog


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


class TestCreditApplication:
    def test_create_application(self, db_session) -> None:
        app = CreditApplication(
            applicant_id="test-001",
            risk_score=0.3,
            risk_category="low",
            default_probability=0.3,
        )
        db_session.add(app)
        db_session.commit()
        result = db_session.query(CreditApplication).filter_by(applicant_id="test-001").first()
        assert result is not None
        assert result.risk_score == pytest.approx(0.3)
        assert result.risk_category == "low"

    def test_multiple_applications(self, db_session) -> None:
        for i in range(5):
            app = CreditApplication(
                applicant_id=f"app-{i}",
                risk_score=float(i) / 10,
                risk_category="low",
                default_probability=float(i) / 10,
            )
            db_session.add(app)
        db_session.commit()
        count = db_session.query(CreditApplication).count()
        assert count == 5

    def test_application_has_id_after_commit(self, db_session) -> None:
        app = CreditApplication(
            applicant_id="test-id",
            risk_score=0.5,
            risk_category="medium",
            default_probability=0.5,
        )
        db_session.add(app)
        db_session.commit()
        assert app.id is not None

    def test_query_by_risk_category(self, db_session) -> None:
        for cat in ["low", "medium", "high"]:
            db_session.add(
                CreditApplication(
                    applicant_id=f"app-{cat}",
                    risk_score=0.3,
                    risk_category=cat,
                    default_probability=0.3,
                )
            )
        db_session.commit()
        low_count = db_session.query(CreditApplication).filter_by(risk_category="low").count()
        assert low_count == 1


class TestDriftLog:
    def test_create_drift_log(self, db_session) -> None:
        log = DriftLog(
            feature_name="debt_to_income",
            ks_statistic=0.15,
            p_value=0.03,
            drift_detected=True,
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(DriftLog).first()
        assert result is not None
        assert result.drift_detected is True
        assert result.ks_statistic == pytest.approx(0.15)

    def test_drift_log_no_drift(self, db_session) -> None:
        log = DriftLog(
            feature_name="annual_income",
            ks_statistic=0.02,
            p_value=0.8,
            drift_detected=False,
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(DriftLog).filter_by(feature_name="annual_income").first()
        assert result.drift_detected is False
