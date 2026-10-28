import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./news_guard.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(String(64), index=True)
    headline = Column(Text)
    source_domain = Column(String(255))
    prediction = Column(String(16))
    confidence = Column(Float)
    credibility_score = Column(Float)
    feature_hash = Column(String(64))
    model_version = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(64))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)
    checked_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_prediction(
    db: Session,
    article_id: str,
    headline: str,
    source_domain: str,
    prediction: str,
    confidence: float,
    credibility_score: float,
    feature_hash: str,
    model_version: str = "1.0.0",
) -> PredictionLog:
    record = PredictionLog(
        article_id=article_id,
        headline=headline,
        source_domain=source_domain,
        prediction=prediction,
        confidence=confidence,
        credibility_score=credibility_score,
        feature_hash=feature_hash,
        model_version=model_version,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
