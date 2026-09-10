"""Seed the database with synthetic prediction records for testing.

Useful for warming up the drift monitor and prediction stats endpoints
after a fresh deployment without needing real traffic.

Usage:
    DATABASE_URL=sqlite:///./property_sage.db python scripts/seed_db.py
"""

from __future__ import annotations

import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, init_db
from app.features import NEIGHBORHOODS, PROPERTY_TYPES, generate_synthetic_data
from app.model import load_models, predict
from app.monitoring import log_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed(n: int = 100) -> None:
    """Generate and persist n synthetic prediction records.

    Args:
        n: Number of prediction records to create.
    """
    init_db()
    price_model, rental_model = load_models()
    X, _, _ = generate_synthetic_data(n=n, seed=777)
    db = SessionLocal()
    try:
        for _, row in X.iterrows():
            data = row.to_dict()
            from app.features import property_to_dataframe
            df = property_to_dataframe(data)
            output = predict(price_model, rental_model, df)
            log_prediction(db, str(uuid.uuid4()), data, output)
        logger.info("Seeded %d prediction records", n)
    finally:
        db.close()


if __name__ == "__main__":
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed(n_records)
