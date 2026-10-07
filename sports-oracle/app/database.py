"""SQLAlchemy ORM models and session management."""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sports_oracle.db")

_engine = None


def get_engine():
    """Lazily create the database engine."""
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
        _engine = create_engine(DATABASE_URL, connect_args=connect_args)
    return _engine


class Base(DeclarativeBase):
    pass


class Match(Base):
    """Historical match records."""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(String(64), unique=True, index=True)
    home_team = Column(String(64), nullable=False)
    away_team = Column(String(64), nullable=False)
    competition = Column(String(64))
    match_date = Column(DateTime)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    outcome = Column(String(4))  # H, D, A
    home_form = Column(Float)
    away_form = Column(Float)
    home_attack = Column(Float)
    away_attack = Column(Float)
    home_defense = Column(Float)
    away_defense = Column(Float)
    h2h_home_wins = Column(Integer, default=0)
    h2h_draws = Column(Integer, default=0)
    h2h_away_wins = Column(Integer, default=0)
    home_rest_days = Column(Integer, default=7)
    away_rest_days = Column(Integer, default=7)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Logged API predictions for monitoring."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), index=True)
    home_team = Column(String(64))
    away_team = Column(String(64))
    competition = Column(String(64))
    features_json = Column(JSON)
    predicted_outcome = Column(String(4))
    prob_home = Column(Float)
    prob_draw = Column(Float)
    prob_away = Column(Float)
    confidence = Column(Float)
    model_version = Column(String(32))
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    """KS-test drift monitoring records."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(64))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Boolean)
    window_size = Column(Integer)
    checked_at = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    """Training run metrics history."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True)
    auc_mean = Column(Float)
    auc_std = Column(Float)
    accuracy = Column(Float)
    n_features = Column(Integer)
    n_samples = Column(Integer)
    trained_at = Column(DateTime, default=datetime.utcnow)
    metrics_json = Column(JSON)


def create_tables() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=get_engine())


def get_session() -> Session:
    """Return a new database session."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return SessionLocal()


def get_all_matches(limit: int = 100) -> list[dict]:
    """Return recent match records for retraining data."""
    session = get_session()
    try:
        rows = session.query(Match).order_by(Match.created_at.desc()).limit(limit).all()
        return [
            {
                "match_id": r.match_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "outcome": r.outcome,
                "home_form": r.home_form,
                "away_form": r.away_form,
            }
            for r in rows
        ]
    except Exception:
        return []
    finally:
        session.close()
