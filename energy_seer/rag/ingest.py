"""Ingest energy consumption patterns into FAISS vector store."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INDEX_PATH = Path("rag/energy_patterns.index")
META_PATH = Path("rag/energy_patterns_meta.json")


def build_pattern_vector(reading: dict) -> np.ndarray:
    """Convert a reading dict into a fixed-length feature vector for FAISS."""
    return np.array(
        [
            float(reading.get("consumption_kwh", 0)),
            float(reading.get("temperature_c", 20)),
            float(reading.get("humidity_pct", 50)),
            float(reading.get("hour_of_day", 12)),
            float(reading.get("day_of_week", 0)),
            float(reading.get("is_holiday", 0)),
        ],
        dtype=np.float32,
    )


def ingest_patterns(patterns: list[dict]) -> int:
    """Ingest consumption patterns into the FAISS index."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss not installed — skipping ingest")
        return 0

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.stack([build_pattern_vector(p) for p in patterns])
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(patterns))
    logger.info("Ingested %d patterns into FAISS", len(patterns))
    return len(patterns)
