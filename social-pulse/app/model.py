"""ML ensemble training and prediction for viral content scoring."""

from __future__ import annotations

import logging
import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from app.features import TextStatTransformer, build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/tmp/social_pulse_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_MODEL_PATH = MODEL_DIR / "current_model.pkl"
CURRENT_PIPELINE_PATH = MODEL_DIR / "current_pipeline.pkl"


@dataclass
class TrainResult:
    """Results of a single training run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_version: str = "1.0.0"
    auc_roc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    cv_auc_scores: list[float] = field(default_factory=list)
    train_samples: int = 0
    feature_names: list[str] = field(default_factory=list)
    duration_sec: float = 0.0


def _build_base_estimators() -> list[tuple[str, Any]]:
    """Return estimators that are available in the current environment."""
    estimators: list[tuple[str, Any]] = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "lr",
            LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
            ),
        ),
    ]
    if _HAS_LGBM:
        estimators.append(
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=63,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    if _HAS_XGB:
        estimators.append(
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    scale_pos_weight=3,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
            )
        )
    return estimators


def _to_dense(X: Any) -> np.ndarray:
    """Convert sparse matrix to dense array if needed."""
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def train(
    samples: list[dict[str, Any]],
    labels: list[int],
    n_folds: int = 5,
    model_version: str = "1.0.0",
) -> TrainResult:
    """Train the ensemble model and persist it to disk.

    Args:
        samples: List of feature dicts, each matching the PostFeatures schema.
        labels: Binary viral labels (1 = viral, 0 = not viral).
        n_folds: Number of stratified CV folds for evaluation.
        model_version: Version string written to the returned result.

    Returns:
        TrainResult with evaluation metrics and metadata.
    """
    t0 = time.time()
    result = TrainResult(model_version=model_version, train_samples=len(samples))

    pipeline = build_feature_pipeline(tfidf_features=200)
    X = pipeline.fit_transform(samples)
    X_dense = _to_dense(X)
    y = np.array(labels)

    estimators = _build_base_estimators()
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_dense, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    result.cv_auc_scores = cv_scores.tolist()
    logger.info("CV AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # Final fit on all data
    ensemble.fit(X_dense, y)

    proba = ensemble.predict_proba(X_dense)[:, 1]
    preds = (proba >= 0.5).astype(int)

    result.auc_roc = roc_auc_score(y, proba)
    result.f1 = f1_score(y, preds, zero_division=0)
    result.precision = precision_score(y, preds, zero_division=0)
    result.recall = recall_score(y, preds, zero_division=0)
    result.duration_sec = time.time() - t0

    try:
        stat_transformer = TextStatTransformer()
        result.feature_names = stat_transformer.FEATURE_NAMES
    except Exception:
        result.feature_names = []

    # Persist
    with open(CURRENT_PIPELINE_PATH, "wb") as fh:
        pickle.dump(pipeline, fh)
    with open(CURRENT_MODEL_PATH, "wb") as fh:
        pickle.dump(ensemble, fh)

    logger.info(
        "Training done in %.1fs — AUC %.4f, F1 %.4f",
        result.duration_sec,
        result.auc_roc,
        result.f1,
    )
    return result


@dataclass
class PredictResult:
    """Output of a single prediction call."""

    viral_probability: float
    viral_label: bool
    ensemble_confidence: float
    model_version: str
    latency_ms: float
    similar_posts: list[dict[str, Any]] = field(default_factory=list)


def load_model() -> tuple[Any, Any] | None:
    """Load the current model and pipeline from disk.

    Returns None when no model has been trained yet.
    """
    if not CURRENT_MODEL_PATH.exists() or not CURRENT_PIPELINE_PATH.exists():
        return None
    try:
        with open(CURRENT_PIPELINE_PATH, "rb") as fh:
            pipeline = pickle.load(fh)
        with open(CURRENT_MODEL_PATH, "rb") as fh:
            model = pickle.load(fh)
        return pipeline, model
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None


def predict(
    sample: dict[str, Any],
    pipeline: Any,
    model: Any,
    model_version: str = "1.0.0",
) -> PredictResult:
    """Run inference on a single sample.

    Args:
        sample: Feature dict matching the PostFeatures schema.
        pipeline: Fitted FeatureUnion pipeline.
        model: Fitted VotingClassifier.
        model_version: Version label for the result.

    Returns:
        PredictResult with probability, label, and confidence.
    """
    t0 = time.perf_counter()
    X = pipeline.transform([sample])
    X_dense = _to_dense(X)
    proba = model.predict_proba(X_dense)[0, 1]
    label = bool(proba >= 0.5)

    # Ensemble confidence: distance from decision boundary
    confidence = float(abs(proba - 0.5) * 2.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictResult(
        viral_probability=float(proba),
        viral_label=label,
        ensemble_confidence=confidence,
        model_version=model_version,
        latency_ms=latency_ms,
    )


def generate_synthetic_data(n: int = 500) -> tuple[list[dict[str, Any]], list[int]]:
    """Generate synthetic training data for bootstrapping the model.

    Viral posts tend to be shorter, have more hashtags, be posted in
    evening hours, and come from accounts with larger followings.
    """
    rng = np.random.default_rng(42)
    samples: list[dict[str, Any]] = []
    labels: list[int] = []

    for i in range(n):
        viral = int(rng.binomial(1, 0.25))
        hashtags = int(rng.poisson(3 if viral else 1))
        mentions = int(rng.poisson(1 if viral else 0.3))
        followers = int(rng.lognormal(7 if viral else 5, 1.5))
        hour = int(rng.choice([18, 19, 20, 21] if viral else range(24)))
        dow = int(rng.integers(0, 7))
        text_length = int(rng.normal(80 if viral else 140, 40))
        text_length = max(10, min(280, text_length))

        tags = " ".join(f"#trending{j}" for j in range(hashtags))
        mentions_str = " ".join(f"@user{j}" for j in range(mentions))
        text = f"Check this out! {tags} {mentions_str} " + "word " * (text_length // 5)
        text = text[:280]

        samples.append(
            {
                "text": text,
                "author_followers": followers,
                "has_media": bool(rng.binomial(1, 0.6 if viral else 0.3)),
                "has_link": bool(rng.binomial(1, 0.4 if viral else 0.2)),
                "hashtag_count": hashtags,
                "mention_count": mentions,
                "hour_of_day": hour,
                "day_of_week": dow,
            }
        )
        labels.append(viral)

    return samples, labels
