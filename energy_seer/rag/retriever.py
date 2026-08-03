"""Retrieve similar energy consumption patterns using FAISS nearest-neighbour search."""

from __future__ import annotations

import logging

import numpy as np

from rag.index import load_index
from rag.ingest import build_pattern_vector

logger = logging.getLogger(__name__)

_index = None
_meta: list[dict] = []


def _ensure_loaded() -> bool:
    global _index, _meta
    if _index is None:
        _index, _meta = load_index()
    return _index is not None


def find_similar_patterns(reading: dict, top_k: int = 5) -> list[dict]:
    """Return the top-k most similar historical energy patterns for a reading."""
    if not _ensure_loaded():
        return []

    vec = build_pattern_vector(reading).reshape(1, -1)
    distances, indices = _index.search(vec, min(top_k, _index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0], strict=False):
        if idx < 0 or idx >= len(_meta):
            continue
        pattern = dict(_meta[idx])
        pattern["similarity_distance"] = round(float(dist), 4)
        results.append(pattern)

    return results


def get_baseline_consumption(
    hour: int,
    day_of_week: int,
    building_type: str = "residential",
) -> float:
    """Return average kWh for similar time-of-day and building type from history."""
    if not _ensure_loaded() or not _meta:
        return 3.5

    similar = [
        m["consumption_kwh"]
        for m in _meta
        if m.get("hour_of_day") == hour
        and m.get("day_of_week") == day_of_week
        and m.get("building_type", "residential") == building_type
    ]
    return round(float(np.mean(similar)), 4) if similar else 3.5
