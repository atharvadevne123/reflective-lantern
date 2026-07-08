"""Seed the database with synthetic manufacturing sensor data for development."""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def seed(n_rows: int = 500) -> None:
    """Generate synthetic predictions and insert them into the database."""
    from app.database import SessionLocal, init_db
    from app.features import generate_synthetic_data
    from app.monitoring import log_prediction

    init_db()
    df = generate_synthetic_data(n_samples=n_rows, seed=999)
    db = SessionLocal()

    try:
        for _, row in df.iterrows():
            sensor = {k: float(row[k]) for k in df.columns if k != "defect"}
            log_prediction(
                db=db,
                sensor_data=sensor,
                prediction=int(row["defect"]),
                defect_probability=float(row["defect"]) * 0.85 + 0.05,
            )
        logger.info("Seeded %d rows into prediction_logs.", n_rows)
    finally:
        db.close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    seed(n)
