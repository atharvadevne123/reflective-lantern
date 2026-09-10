"""Backfill FAISS index from historical prediction records in the database."""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.database import MarketRegimePrediction, SessionLocal
from app.model import FAISS_INDEX_PATH, build_faiss_index

logger = logging.getLogger(__name__)


def backfill(limit: int = 5000) -> int:
    """Build FAISS index from the most recent `limit` DB prediction records.

    Returns the number of vectors added to the index.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(MarketRegimePrediction)
            .filter(MarketRegimePrediction.volatility.isnot(None))
            .order_by(MarketRegimePrediction.created_at.desc())
            .limit(limit)
            .all()
        )
        if not rows:
            logger.info("No prediction records found; skipping FAISS backfill")
            return 0

        embeddings = np.array(
            [
                [
                    r.volatility or 0.0,
                    r.momentum or 0.0,
                    r.volume_ratio or 1.0,
                    r.correlation or 0.0,
                    r.beta or 1.0,
                ]
                for r in rows
            ],
            dtype=np.float32,
        )
        build_faiss_index(embeddings)

        metadata = [
            {
                "id": r.id,
                "ticker": r.ticker,
                "regime": r.regime,
                "date": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
        meta_path = FAISS_INDEX_PATH.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        logger.info("FAISS backfill complete: %d vectors", len(rows))
        return len(rows)
    except Exception as e:
        logger.error("FAISS backfill failed: %s", e)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = backfill()
    print(f"Backfilled {n} vectors into FAISS index")
