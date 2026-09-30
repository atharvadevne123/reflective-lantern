"""SQLAlchemy models and session management."""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sport_cast.db")

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
        _engine = create_engine(DATABASE_URL, connect_args=connect_args)
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


class Base(DeclarativeBase):
    pass


class MatchRecord(Base):
    __tablename__ = "match_records"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(String(64), unique=True, index=True)
    home_team = Column(String(128))
    away_team = Column(String(128))
    sport = Column(String(64), default="football")
    match_date = Column(DateTime, nullable=True)
    result = Column(String(16), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(String(64), index=True)
    home_team = Column(String(128))
    away_team = Column(String(128))
    predicted_outcome = Column(String(16))
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    confidence = Column(Float)
    model_version = Column(String(32), default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(128))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Boolean)
    checked_at = Column(DateTime, default=datetime.utcnow)


class PlayerPerformance(Base):
    __tablename__ = "player_performances"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String(64), index=True)
    player_name = Column(String(128))
    team = Column(String(128))
    sport = Column(String(64), default="football")
    performance_score = Column(Float)
    fatigue_index = Column(Float)
    form_rating = Column(Float)
    raw_features = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:  # type: ignore[return]
    db = _get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=_get_engine())
