"""ML model training, inference, and FAISS item-similarity index."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.features import (
    INTERACTION_COLS,
    ITEM_COLS,
    USER_COLS,
    build_feature_pipeline,
    make_sample_dataframe,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index.idx"))
ITEM_IDS_PATH = Path(os.getenv("ITEM_IDS_PATH", "item_ids.json"))

_FEATURE_COLS = USER_COLS + ITEM_COLS + INTERACTION_COLS


def _build_ensemble() -> VotingClassifier:
    lgbm = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        verbosity=0,
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    return VotingClassifier(
        estimators=[("lgbm", lgbm), ("xgb", xgb), ("rf", rf)],
        voting="soft",
    )


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict[str, Any]]:
    """Train the intent prediction pipeline with 5-fold CV and return metrics."""
    feat_pipe = build_feature_pipeline()
    ensemble = _build_ensemble()

    full_pipeline = Pipeline([("features", feat_pipe), ("model", ensemble)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    full_pipeline.fit(X, y)

    metrics: dict[str, Any] = {
        "auc_mean": round(float(scores.mean()), 4),
        "auc_std": round(float(scores.std()), 4),
        "n_features": X.shape[1],
        "n_samples": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "model_version": "1.0.0",
    }

    joblib.dump(full_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Model trained: AUC=%.4f±%.4f", metrics["auc_mean"], metrics["auc_std"])
    return full_pipeline, metrics


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        logger.warning("No saved model found; training bootstrap model.")
        df = make_sample_dataframe(n=500)
        X = df[_FEATURE_COLS]
        rng = np.random.default_rng(42)
        y = pd.Series(rng.integers(0, 2, len(df)), name="purchased")
        pipe, _ = train_model(X, y)
        return pipe
    return joblib.load(MODEL_PATH)


def predict_intent(pipeline: Pipeline, X: pd.DataFrame) -> tuple[list[float], list[int]]:
    """Return (probabilities, predicted_labels) for purchase intent."""
    proba = pipeline.predict_proba(X)[:, 1]
    labels = (proba >= 0.5).astype(int)
    return proba.tolist(), labels.tolist()


# ---------------------------------------------------------------------------
# FAISS-based item similarity index
# ---------------------------------------------------------------------------


def build_faiss_index(item_vectors: np.ndarray, item_ids: list[str]) -> Any:
    """Build a FAISS flat-L2 index from item embedding vectors."""
    try:
        import faiss  # type: ignore

        d = item_vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(item_vectors.astype(np.float32))
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        ITEM_IDS_PATH.write_text(json.dumps(item_ids))
        logger.info("FAISS index built: %d items, dim=%d", len(item_ids), d)
        return index
    except ImportError:
        logger.warning("FAISS not available; using brute-force fallback.")
        return _BruteForceIndex(item_vectors, item_ids)


def load_faiss_index() -> tuple[Any, list[str]]:
    """Load the FAISS index and corresponding item IDs."""
    try:
        import faiss  # type: ignore

        if FAISS_INDEX_PATH.exists() and ITEM_IDS_PATH.exists():
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            item_ids = json.loads(ITEM_IDS_PATH.read_text())
            return index, item_ids
    except ImportError:
        pass

    logger.info("Generating synthetic FAISS index.")
    rng = np.random.default_rng(0)
    vecs = rng.random((200, 32)).astype(np.float32)
    ids = [f"item_{i:04d}" for i in range(200)]
    index = build_faiss_index(vecs, ids)
    return index, ids


def search_similar_items(
    index: Any, item_ids: list[str], query_vector: np.ndarray, top_k: int = 5
) -> list[dict[str, Any]]:
    """Return top-k most similar items to the query vector."""
    query = query_vector.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query, top_k + 1)
    results = []
    for dist, idx in zip(distances[0], indices[0], strict=False):
        if 0 <= idx < len(item_ids):
            results.append(
                {"item_id": item_ids[idx], "similarity_score": round(1 / (1 + float(dist)), 4)}
            )
    return results[:top_k]


class _BruteForceIndex:
    """Brute-force cosine-similarity fallback when FAISS is unavailable."""

    def __init__(self, vectors: np.ndarray, item_ids: list[str]) -> None:
        self._vecs = vectors.astype(np.float32)
        self._ids = item_ids

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        diffs = self._vecs - query
        dists = np.sum(diffs**2, axis=1)
        top_idx = np.argsort(dists)[:k]
        return dists[top_idx].reshape(1, -1), top_idx.reshape(1, -1)

    def add(self, vectors: np.ndarray) -> None:
        self._vecs = np.vstack([self._vecs, vectors.astype(np.float32)])
