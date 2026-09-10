"""ML model: XGBoost/LightGBM ensemble for market regime detection."""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/tmp/signal_forge_model.pkl"))
FAISS_INDEX_PATH = Path(os.environ.get("FAISS_INDEX_PATH", "/tmp/signal_forge.faiss"))

REGIMES = ["bull", "bear", "sideways", "volatile"]


@dataclass
class PredictionResult:
    """Structured prediction output."""

    ticker: str
    regime: str
    confidence: float
    risk_score: float
    volatility: float
    momentum: float
    correlation: float
    volume_ratio: float
    beta: float
    similar_periods: list[dict[str, Any]]


def _build_ensemble() -> VotingClassifier:
    """Construct the XGBoost + LightGBM + RandomForest ensemble."""
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("rf", rf_clf)],
        voting="soft",
    )


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> tuple[VotingClassifier, dict[str, float]]:
    """Train ensemble with stratified k-fold CV and return model + metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        fold_model = _build_ensemble()
        fold_model.fit(X_train, y_train)
        proba = fold_model.predict_proba(X_val)
        try:
            auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
            auc_scores.append(auc)
        except Exception as e:
            logger.warning("AUC computation failed on fold %d: %s", fold, e)

    final_model = _build_ensemble()
    final_model.fit(X, y)
    metrics = {
        "cv_auc_mean": float(np.mean(auc_scores)) if auc_scores else 0.0,
        "cv_auc_std": float(np.std(auc_scores)) if auc_scores else 0.0,
    }
    logger.info("Training complete: %s", metrics)
    return final_model, metrics


def save_model(model: VotingClassifier) -> None:
    """Persist ensemble model to disk."""
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", MODEL_PATH)
    except Exception as e:
        logger.error("Failed to save model: %s", e)
        raise


def load_model() -> VotingClassifier:
    """Load ensemble model from disk, or create synthetic model if absent."""
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info("Model loaded from %s", MODEL_PATH)
            return model
        except Exception as e:
            logger.warning("Failed to load model, creating synthetic: %s", e)
    return _create_synthetic_model()


def _create_synthetic_model() -> VotingClassifier:
    """Create and train a model on synthetic data for cold-start."""
    rng = np.random.default_rng(42)
    X_syn = rng.standard_normal((400, 5))
    y_syn = rng.integers(0, 4, size=400)
    model = _build_ensemble()
    model.fit(X_syn, y_syn)
    save_model(model)
    return model


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat L2 FAISS index from feature embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


def load_faiss_index() -> faiss.Index | None:
    """Load FAISS index from disk if available."""
    if FAISS_INDEX_PATH.exists():
        try:
            return faiss.read_index(str(FAISS_INDEX_PATH))
        except Exception as e:
            logger.warning("Failed to load FAISS index: %s", e)
    return None


def find_similar_periods(
    query: np.ndarray,
    index: faiss.Index | None,
    metadata: list[dict[str, Any]],
    k: int = 3,
) -> list[dict[str, Any]]:
    """Search FAISS index for k most similar historical market periods."""
    if index is None or index.ntotal == 0:
        return []
    try:
        q = query.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(q, min(k, index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(metadata):
                results.append({**metadata[idx], "distance": float(dist)})
        return results
    except Exception as e:
        logger.error("FAISS search failed: %s", e)
        return []


def compute_risk_score(
    volatility: float,
    momentum: float,
    beta: float,
    regime: str,
) -> float:
    """Compute a 0-1 portfolio risk score from key factors."""
    vol_score = min(volatility / 0.5, 1.0)
    mom_score = min(abs(momentum) / 0.3, 1.0)
    beta_score = min(abs(beta) / 2.0, 1.0)
    regime_score = {"bull": 0.2, "bear": 0.85, "volatile": 0.9, "sideways": 0.4}.get(
        regime, 0.5
    )
    return round(0.3 * vol_score + 0.2 * mom_score + 0.2 * beta_score + 0.3 * regime_score, 4)


def predict(
    features: np.ndarray,
    ticker: str,
    raw_values: dict[str, float],
) -> PredictionResult:
    """Run full inference: ensemble predict + risk score + FAISS lookup."""
    model = load_model()
    faiss_index = load_faiss_index()
    metadata_path = FAISS_INDEX_PATH.with_suffix(".meta.json")
    metadata: list[dict[str, Any]] = []
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning("Failed to load FAISS metadata: %s", e)

    feat_vec = features[-1:] if features.ndim == 2 else features.reshape(1, -1)
    proba = model.predict_proba(feat_vec)[0]
    regime_idx = int(np.argmax(proba))
    regime = REGIMES[regime_idx]
    confidence = round(float(proba[regime_idx]), 4)

    vol = raw_values.get("volatility", 0.0)
    mom = raw_values.get("momentum", 0.0)
    beta = raw_values.get("beta", 1.0)
    risk_score = compute_risk_score(vol, mom, beta, regime)
    similar = find_similar_periods(feat_vec[0], faiss_index, metadata)

    return PredictionResult(
        ticker=ticker,
        regime=regime,
        confidence=confidence,
        risk_score=risk_score,
        volatility=vol,
        momentum=mom,
        correlation=raw_values.get("correlation", 0.0),
        volume_ratio=raw_values.get("volume_ratio", 1.0),
        beta=beta,
        similar_periods=similar,
    )
