"""Tests for ticket-oracle app/database.py."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, PredictionLog


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def db_session(engine):
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with factory() as session:
        yield session


class TestSchema:
    def test_prediction_logs_table(self, engine) -> None:
        assert "prediction_logs" in inspect(engine).get_table_names()


class TestPredictionLog:
    def test_insert(self, db_session: Session) -> None:
        log = PredictionLog(
            ticket_id="TKT-001",
            priority_predicted="P2",
            priority_p1_prob=0.05,
            priority_p2_prob=0.75,
            priority_p3_prob=0.15,
            priority_p4_prob=0.05,
            sla_breach_risk=0.3,
            sla_breach_predicted=False,
            estimated_resolution_hours=4.0,
            confidence=0.85,
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None
        assert result.ticket_id == "TKT-001"

    def test_query_by_ticket_id(self, db_session: Session) -> None:
        for i in range(3):
            db_session.add(
                PredictionLog(
                    ticket_id=f"TKT-{i:03d}",
                    priority_predicted="P3",
                    priority_p1_prob=0.05,
                    priority_p2_prob=0.1,
                    priority_p3_prob=0.7,
                    priority_p4_prob=0.15,
                    sla_breach_risk=0.1,
                    sla_breach_predicted=False,
                    estimated_resolution_hours=8.0,
                    confidence=0.75,
                )
            )
        db_session.commit()
        assert db_session.query(PredictionLog).count() == 3

    @pytest.mark.parametrize("priority", ["P1", "P2", "P3", "P4"])
    def test_all_priority_values(self, db_session: Session, priority: str) -> None:
        log = PredictionLog(
            ticket_id=f"TKT-{priority}",
            priority_predicted=priority,
            priority_p1_prob=0.25,
            priority_p2_prob=0.25,
            priority_p3_prob=0.25,
            priority_p4_prob=0.25,
            sla_breach_risk=0.5,
            sla_breach_predicted=False,
            estimated_resolution_hours=6.0,
            confidence=0.6,
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).filter_by(priority_predicted=priority).first()
        assert result is not None
