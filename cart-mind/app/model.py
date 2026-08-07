"""Model training, inference, and the FAISS item-similarity index.

The intent model is a soft-voting ensemble of LightGBM, XGBoost, and a random
forest, wrapped in a single sklearn ``Pipeline`` together with the feature
stages. Keeping the transforms inside the fitted pipeline means training and
serving cannot drift apart: there is exactly one transform path, and it is the
one that gets pickled.
"""

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

# Dimensionality of item embeddings. Build and query paths must agree on this,
# so it lives in one place rather than being repeated as a literal.
EMBED_DIM = int(os.getenv("EMBED_DIM", "32"))

_FEATURE_COLS = USER_COLS + ITEM_COLS + INTERACTION_COLS


def _build_ensemble() -> VotingClassifier:
    """Build the soft-voting ensemble.

    Soft voting averages predicted probabilities rather than taking a majority
    of hard labels, which preserves the calibration the ranking endpoints need —
    ``/recommend`` sorts by score, so a well-ordered probability matters more
    than the 0.5 threshold decision.

    Returns:
        An unfitted ``VotingClassifier`` over LightGBM, XGBoost, and a forest.
    """
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
    """Train the intent pipeline, score it with 5-fold CV, and persist it.

    Cross-validation runs on the full pipeline rather than the bare estimator, so
    the feature transforms are re-fitted inside each fold and the AUC does not
    inherit leakage from transforms fitted on the whole dataset.

    Note:
        Writes ``MODEL_PATH`` and ``METRICS_PATH`` as a side effect. Callers that
        need the incumbent's metrics must read them *before* calling this.

    Args:
        X: Raw feature frame; transforms are applied inside the pipeline.
        y: Binary purchase labels.

    Returns:
        The fitted pipeline paired with its cross-validated metrics.
    """
    feat_pipe = build_feature_pipeline()
    ensemble = _build_ensemble()

    full_pipeline = Pipeline([("features", feat_pipe), ("model", ensemble)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    full_pipeline.fit(X, y)

    model_version = os.getenv("MODEL_VERSION", "1.0.0")
    metrics: dict[str, Any] = {
        "auc_mean": round(float(scores.mean()), 4),
        "auc_std": round(float(scores.std()), 4),
        "auc_min": round(float(scores.min()), 4),
        "auc_max": round(float(scores.max()), 4),
        "n_features": X.shape[1],
        "n_samples": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "model_version": model_version,
    }

    joblib.dump(full_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info(
        "Model trained: AUC=%.4f±%.4f (min=%.4f max=%.4f) n=%d",
        metrics["auc_mean"],
        metrics["auc_std"],
        metrics["auc_min"],
        metrics["auc_max"],
        metrics["n_samples"],
    )
    return full_pipeline, metrics


def load_model() -> Pipeline:
    """Load the persisted pipeline, training a bootstrap model if none exists.

    The bootstrap path exists so a fresh checkout serves traffic immediately. It
    fits on random labels and is therefore near-chance by construction — run
    ``scripts/train.py`` to replace it with a model that has learned something.

    Returns:
        A fitted pipeline ready for inference.
    """
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
    """Score rows for purchase intent.

    Args:
        pipeline: Fitted pipeline from :func:`train_model` or :func:`load_model`.
        X: Raw feature frame with one row per user-item pair.

    Returns:
        Positive-class probabilities and their 0.5-thresholded labels.
    """
    proba = pipeline.predict_proba(X)[:, 1]
    labels = (proba >= 0.5).astype(int)
    return proba.tolist(), labels.tolist()


# ---------------------------------------------------------------------------
# FAISS-based item similarity index
# ---------------------------------------------------------------------------


def build_faiss_index(item_vectors: np.ndarray, item_ids: list[str], persist: bool = False) -> Any:
    """Build a FAISS flat-L2 index from item embedding vectors.

    Persistence is opt-in. Writing to the shared index path by default meant any
    caller building a throwaway index silently replaced the deployed serving
    index — including tests, which then left a mismatched-dimension index on disk
    for the API to trip over.

    Args:
        item_vectors: Item embeddings, shape ``(n_items, dim)``.
        item_ids: Item IDs positionally aligned to ``item_vectors``.
        persist: Write the index and IDs to disk, replacing the serving index.

    Returns:
        A FAISS index, or a :class:`_BruteForceIndex` if FAISS is unavailable.
    """
    try:
        import faiss  # type: ignore

        d = item_vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(item_vectors.astype(np.float32))
        if persist:
            faiss.write_index(index, str(FAISS_INDEX_PATH))
            ITEM_IDS_PATH.write_text(json.dumps(item_ids))
            logger.info("FAISS index persisted to %s", FAISS_INDEX_PATH)
        logger.info("FAISS index built: %d items, dim=%d", len(item_ids), d)
        return index
    except ImportError:
        logger.warning("FAISS not available; using brute-force fallback.")
        return _BruteForceIndex(item_vectors, item_ids)


def load_faiss_index() -> tuple[Any, list[str]]:
    """Load the persisted FAISS index, or build a synthetic one as a fallback.

    A persisted index is only accepted if its dimensionality matches
    :data:`EMBED_DIM`. A stale index left behind at a different dimension would
    otherwise be loaded happily and then fail at query time, deep inside FAISS,
    with an opaque assertion — so it is rejected here and rebuilt instead.

    Returns:
        The loaded or freshly built index, paired with its item IDs.
    """
    try:
        import faiss  # type: ignore

        if FAISS_INDEX_PATH.exists() and ITEM_IDS_PATH.exists():
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            if index.d == EMBED_DIM:
                item_ids = json.loads(ITEM_IDS_PATH.read_text())
                logger.info("Loaded FAISS index: %d items, dim=%d", index.ntotal, index.d)
                return index, item_ids
            logger.warning(
                "Discarding persisted index: dim=%d, expected %d. Rebuilding.",
                index.d,
                EMBED_DIM,
            )
    except ImportError:
        pass

    logger.info("Generating synthetic FAISS index.")
    rng = np.random.default_rng(0)
    vecs = rng.random((200, EMBED_DIM)).astype(np.float32)
    ids = [f"item_{i:04d}" for i in range(200)]
    index = build_faiss_index(vecs, ids)
    return index, ids


def search_similar_items(
    index: Any, item_ids: list[str], query_vector: np.ndarray, top_k: int = 5
) -> list[dict[str, Any]]:
    """Return the top-k items nearest to the query vector.

    Args:
        index: A FAISS index or :class:`_BruteForceIndex`.
        item_ids: Item IDs positionally aligned to the index contents.
        query_vector: Query embedding; must match the index dimensionality.
        top_k: Number of neighbours to return.

    Returns:
        Neighbours as ``{"item_id", "similarity_score"}``, nearest first.

    Raises:
        ValueError: If the query dimensionality does not match the index.
    """
    query = query_vector.astype(np.float32).reshape(1, -1)
    index_dim = getattr(index, "d", None)
    if index_dim is not None and query.shape[1] != index_dim:
        raise ValueError(
            f"Query dimension {query.shape[1]} does not match index dimension {index_dim}"
        )
    distances, indices = index.search(query, top_k + 1)
    results = []
    for dist, idx in zip(distances[0], indices[0], strict=False):
        if 0 <= idx < len(item_ids):
            results.append(
                {"item_id": item_ids[idx], "similarity_score": round(1 / (1 + float(dist)), 4)}
            )
    return results[:top_k]


class _BruteForceIndex:
    """Exact L2 nearest-neighbour fallback used when FAISS is unavailable.

    Mirrors the slice of the FAISS API this module uses (``search``, ``add``, and
    an absent ``d`` attribute), so the similarity endpoint keeps working on hosts
    where ``faiss-cpu`` will not install. Exact rather than approximate, so it is
    O(n) per query — correct, and fine at catalogue sizes in the low thousands.
    """

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
