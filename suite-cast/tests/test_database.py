"""Tests for suite-cast app/database.py."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, ModelMetrics, Prediction


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


class TestDatabaseSchema:
    def test_predictions_table_exists(self, engine) -> None:
        inspector = inspect(engine)
        assert "predictions" in inspector.get_table_names()

    def test_model_metrics_table_exists(self, engine) -> None:
        inspector = inspect(engine)
        assert "model_metrics" in inspector.get_table_names()


class TestPrediction:
    def test_insert_prediction(self, db_session: Session) -> None:
        pred = Prediction(
            input_features={"rooms": 2},
            demand_score=0.8,
            suggested_rate=200.0,
            model_version="1.0.0",
        )
        db_session.add(pred)
        db_session.commit()
        result = db_session.query(Prediction).first()
        assert result is not None
        assert result.demand_score == pytest.approx(0.8)

    def test_unique_request_ids(self, db_session: Session) -> None:
        for _ in range(3):
            db_session.add(
                Prediction(
                    input_features={},
                    demand_score=0.5,
                    suggested_rate=150.0,
                    model_version="1.0.0",
                )
            )
        db_session.commit()
        ids = [p.request_id for p in db_session.query(Prediction).all()]
        assert len(set(ids)) == 3


class TestModelMetrics:
    def test_insert_metrics(self, db_session: Session) -> None:
        m = ModelMetrics(
            model_version="1.0.0",
            auc_mean=0.92,
            auc_std=0.02,
            n_features=10,
        )
        db_session.add(m)
        db_session.commit()
        result = db_session.query(ModelMetrics).first()
        assert result is not None
        assert result.auc_mean == pytest.approx(0.92)
