"""Seed the database with synthetic prediction records for development and testing."""

from __future__ import annotations

import logging
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.constants import REGIMES
from app.database import MarketRegimePrediction, SessionLocal, create_tables

logger = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "META", "BRK-B"]


def seed(n: int = 200, seed_val: int = 42) -> int:
    """Insert `n` synthetic prediction records into the database.

    Returns the number of records inserted.
    """
    create_tables()
    rng = np.random.default_rng(seed_val)
    db = SessionLocal()
    inserted = 0
    try:
        for _ in range(n):
            regime = random.choice(REGIMES)
            record = MarketRegimePrediction(
                ticker=random.choice(TICKERS),
                regime=regime,
                confidence=float(rng.uniform(0.5, 0.99)),
                risk_score=float(rng.uniform(0.1, 0.95)),
                volatility=float(rng.uniform(0.05, 0.6)),
                momentum=float(rng.uniform(-0.3, 0.3)),
                correlation=float(rng.uniform(-0.5, 0.95)),
                volume_ratio=float(rng.uniform(0.5, 3.0)),
                beta=float(rng.uniform(0.3, 2.5)),
            )
            db.add(record)
            inserted += 1
        db.commit()
        logger.info("Seeded %d synthetic records", inserted)
        return inserted
    except Exception as e:
        db.rollback()
        logger.error("Seed failed: %s", e)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = seed()
    print(f"Seeded {n} synthetic records")
