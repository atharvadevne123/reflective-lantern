"""Train the Cart-Mind purchase-intent model and build the FAISS item index."""

from __future__ import annotations

import json
import logging

import numpy as np

from app.features import (
    INTERACTION_COLS,
    ITEM_COLS,
    USER_COLS,
    make_purchase_labels,
    make_sample_dataframe,
)
from app.model import build_faiss_index, train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = USER_COLS + ITEM_COLS + INTERACTION_COLS
N_SAMPLES = 5000
N_ITEMS = 500
EMBED_DIM = 32


def main() -> None:
    """Train the ensemble, persist artifacts, and build the similarity index."""
    logger.info("Generating %d training rows...", N_SAMPLES)
    df = make_sample_dataframe(n=N_SAMPLES, seed=42)
    X = df[FEATURE_COLS]
    y = make_purchase_labels(df, seed=42)
    logger.info("Positive rate: %.3f", y.mean())

    logger.info("Training ensemble with 5-fold stratified CV...")
    _, metrics = train_model(X, y)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    logger.info("Building FAISS index over %d items...", N_ITEMS)
    rng = np.random.default_rng(7)
    vectors = rng.normal(0, 1, (N_ITEMS, EMBED_DIM)).astype(np.float32)
    item_ids = [f"item_{i:04d}" for i in range(N_ITEMS)]
    build_faiss_index(vectors, item_ids)

    logger.info("Done. Artifacts: model.joblib, metrics.json, faiss_index.idx, item_ids.json")


if __name__ == "__main__":
    main()
