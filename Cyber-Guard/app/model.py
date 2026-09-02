"""ML model training and prediction for network intrusion detection."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.features import FEATURE_NAMES, build_feature_pipeline
from app.storage import upload_artifact
from app.tracking import log_metrics, log_params, track_run

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")

THREAT_CLASSES = [
    "normal",
    "dos",
    "probe",
    "r2l",
    "u2r",
]

label_encoder = LabelEncoder()
label_encoder.fit(THREAT_CLASSES)


def _build_ensemble() -> VotingClassifier:
    """Build the soft-voting XGBoost + RandomForest ensemble."""
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    return VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft",
    )


# Per-class connection profiles, loosely modelled on the KDD Cup taxonomy.
# Each entry gives the sampling distribution of the raw packet fields for that
# class, so the generated features actually carry signal about the label.
# Drawing labels independently of features (as an unconditioned rng.choice
# would) yields an AUC of ~0.5 no matter how good the model is.
_CLASS_PROFILES: dict[str, dict[str, Any]] = {
    # Ordinary request/response: bytes flow both ways, connection completes.
    "normal": {
        "src_scale": 300.0, "dst_scale": 800.0, "dur_scale": 2.0,
        "protocols": (["tcp", "udp"], [0.8, 0.2]),
        "services": (["http", "smtp", "dns", "ssh"], [0.55, 0.2, 0.15, 0.1]),
        "flags": (["SF"], [1.0]),
    },
    # Flood: huge outbound volume, near-zero response, half-open connections.
    "dos": {
        "src_scale": 5000.0, "dst_scale": 5.0, "dur_scale": 0.2,
        "protocols": (["tcp", "icmp"], [0.6, 0.4]),
        "services": (["http", "other"], [0.7, 0.3]),
        "flags": (["S0", "REJ"], [0.7, 0.3]),
    },
    # Scan: tiny payloads, no response, connection refused across services.
    "probe": {
        "src_scale": 40.0, "dst_scale": 8.0, "dur_scale": 0.1,
        "protocols": (["tcp", "icmp"], [0.5, 0.5]),
        "services": (["other", "ftp", "ssh", "dns"], [0.4, 0.2, 0.2, 0.2]),
        "flags": (["REJ", "S0"], [0.6, 0.4]),
    },
    # Remote-to-local: credential pushes against auth-bearing services.
    "r2l": {
        "src_scale": 700.0, "dst_scale": 350.0, "dur_scale": 9.0,
        "protocols": (["tcp"], [1.0]),
        "services": (["ftp", "telnet", "pop_3"], [0.45, 0.35, 0.2]),
        "flags": (["SF", "RSTO"], [0.7, 0.3]),
    },
    # Privilege escalation: long interactive session pulling data back.
    "u2r": {
        "src_scale": 250.0, "dst_scale": 6000.0, "dur_scale": 45.0,
        "protocols": (["tcp"], [1.0]),
        "services": (["telnet", "ssh"], [0.6, 0.4]),
        "flags": (["SF"], [1.0]),
    },
}

CLASS_PRIORS = [0.60, 0.15, 0.10, 0.10, 0.05]


def generate_synthetic_data(
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate labelled connection records with class-conditional signal.

    Each label is drawn first, then its packet fields are sampled from that
    class's profile in :data:`_CLASS_PROFILES`. This is what makes the data
    learnable: sampling features and labels independently would cap AUC at
    chance regardless of the model.

    Args:
        n_samples: Number of connections to generate.
        seed: Seed for the random generator.

    Returns:
        A ``(features, labels)`` pair. Features hold the six raw packet
        columns; labels are drawn from :data:`THREAT_CLASSES`.
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice(THREAT_CLASSES, n_samples, p=CLASS_PRIORS)

    rows = []
    for label in labels:
        p = _CLASS_PROFILES[label]
        protos, proto_w = p["protocols"]
        svcs, svc_w = p["services"]
        flags, flag_w = p["flags"]
        rows.append({
            "src_bytes": float(rng.exponential(p["src_scale"])),
            "dst_bytes": float(rng.exponential(p["dst_scale"])),
            "duration": float(rng.exponential(p["dur_scale"])),
            "protocol_type": str(rng.choice(protos, p=proto_w)),
            "service": str(rng.choice(svcs, p=svc_w)),
            "flag": str(rng.choice(flags, p=flag_w)),
        })

    return pd.DataFrame(rows), pd.Series(labels)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: str = MODEL_PATH,
    metrics_path: str = METRICS_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the ensemble with 5-fold CV and persist model plus metrics.

    Args:
        X: Raw connection records with the six standard packet columns.
        y: Threat-class labels drawn from :data:`THREAT_CLASSES`.
        model_path: Destination for the serialised pipeline.
        metrics_path: Destination for the JSON metrics report.

    Returns:
        A ``(fitted_pipeline, metrics)`` pair. Metrics carry cross-validated
        accuracy and one-vs-rest macro AUC-ROC, each with its fold standard
        deviation.
    """
    feature_pipe = build_feature_pipeline()
    ensemble = _build_ensemble()

    pipe = Pipeline([
        ("features", feature_pipe),
        ("model", ensemble),
    ])

    y_enc = label_encoder.transform(y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring="accuracy")

    # Macro one-vs-rest AUC is the more informative headline here: the class
    # mix is heavily skewed toward `normal`, so accuracy alone would look
    # respectable for a model that never predicts a rare attack class at all.
    # A fold that happens to contain none of a rare class (u2r is 5% of the
    # prior) cannot produce an OvR AUC. sklearn does not raise for this -- it
    # warns and sets that fold's score to NaN -- so the folds that did score
    # are averaged with nanmean, and an all-NaN result becomes None rather
    # than NaN, which json.dump would otherwise emit as invalid JSON.
    auc_scores = cross_val_score(
        pipe, X, y_enc, cv=cv, scoring="roc_auc_ovr_weighted", error_score=np.nan
    )
    scored_folds = int(np.count_nonzero(~np.isnan(auc_scores)))
    if scored_folds == 0:
        logger.warning("AUC unavailable: no CV fold contained every class")
        auc_mean = auc_std = None
    else:
        if scored_folds < len(auc_scores):
            logger.warning(
                "AUC averaged over %d of %d folds; the rest lacked a class",
                scored_folds, len(auc_scores),
            )
        auc_mean = float(np.nanmean(auc_scores))
        auc_std = float(np.nanstd(auc_scores))

    pipe.fit(X, y_enc)

    metrics: dict[str, Any] = {
        "accuracy_mean": float(cv_scores.mean()),
        "accuracy_std": float(cv_scores.std()),
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_scored_folds": scored_folds,
        "n_features": len(FEATURE_NAMES),
        "n_samples": len(X),
        "classes": THREAT_CLASSES,
        "cv_folds": 5,
    }

    with track_run("train_model"):
        log_params({
            "n_estimators": 100,
            "max_depth": 4,
            "ensemble": "xgboost+randomforest",
            "n_features": len(FEATURE_NAMES),
        })
        log_metrics(metrics)

    joblib.dump({"pipeline": pipe, "label_encoder": label_encoder}, model_path)
    with open(metrics_path, "w") as fh:
        # allow_nan=False makes an accidental NaN fail here rather than
        # silently writing a metrics.json no strict JSON parser will read.
        json.dump(metrics, fh, indent=2, allow_nan=False)

    upload_artifact(model_path)

    logger.info(
        "model trained accuracy=%.4f auc=%s n=%d",
        metrics["accuracy_mean"],
        f"{auc_mean:.4f}" if auc_mean is not None else "n/a",
        len(X),
    )
    return pipe, metrics


def load_model(model_path: str = MODEL_PATH) -> tuple[Pipeline, LabelEncoder]:
    """Load a serialised pipeline and its label encoder.

    Args:
        model_path: Path written by :func:`train_model`.

    Returns:
        A ``(pipeline, label_encoder)`` pair.
    """
    artifact = joblib.load(model_path)
    return artifact["pipeline"], artifact["label_encoder"]


def predict(
    X: pd.DataFrame,
    pipeline: Pipeline,
    le: LabelEncoder,
) -> dict[str, Any]:
    """Classify a single connection.

    Args:
        X: A one-row DataFrame of raw connection fields.
        pipeline: A fitted pipeline from :func:`train_model`.
        le: The matching label encoder.

    Returns:
        A dict with ``prediction``, ``confidence`` and the full
        ``class_probabilities`` mapping.
    """
    proba = pipeline.predict_proba(X)[0]
    class_idx = int(np.argmax(proba))
    label = le.inverse_transform([class_idx])[0]
    return {
        "prediction": label,
        "confidence": float(proba[class_idx]),
        "class_probabilities": {cls: float(p) for cls, p in zip(le.classes_, proba)},
    }


def ensure_model_exists(model_path: str = MODEL_PATH) -> None:
    """Train and persist a model on first boot if none exists yet.

    Args:
        model_path: Path checked for an existing artifact.
    """
    if not Path(model_path).exists():
        X, y = generate_synthetic_data(500)
        train_model(X, y, model_path)
