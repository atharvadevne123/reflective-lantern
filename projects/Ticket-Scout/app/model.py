"""ML model training, evaluation, and prediction."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from app.features import TextFeatureTransformer

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
CATEGORY_LABELS = ["access", "email", "hardware", "network", "software"]
MODEL_VERSION = "1.0.0"


def _make_category_pipeline() -> Pipeline:
    return Pipeline([
        ("features", TextFeatureTransformer(max_tfidf_features=200)),
        ("model", LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )),
    ])


def _make_breach_pipeline() -> Pipeline:
    return Pipeline([
        ("features", TextFeatureTransformer(max_tfidf_features=150)),
        ("model", LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=24,
            subsample=0.8,
            scale_pos_weight=2.0,
            random_state=42,
            verbose=-1,
        )),
    ])


def _make_resolution_pipeline() -> Pipeline:
    return Pipeline([
        ("features", TextFeatureTransformer(max_tfidf_features=100)),
        ("model", LGBMRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=24,
            random_state=42,
            verbose=-1,
        )),
    ])


def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate realistic synthetic support ticket data for training.

    Args:
        n_samples: Number of tickets to generate.

    Returns:
        DataFrame with ticket features and labels.
    """
    rng = np.random.default_rng(42)

    category_subjects = {
        "network": [
            "VPN not connecting after update",
            "Internet connection dropping intermittently",
            "Firewall blocking internal application",
            "WiFi password reset needed",
            "Network drive not accessible from home",
        ],
        "hardware": [
            "Laptop screen flickering issue",
            "Keyboard keys not responding",
            "Monitor not detected",
            "Printer jammed and error light on",
            "Mouse double-clicking unintentionally",
        ],
        "software": [
            "Application crashes on startup",
            "Unable to install required software",
            "License expired for design tool",
            "Software update failed with error",
            "Application freezes when saving large files",
        ],
        "access": [
            "Password reset request",
            "Account locked out after failed attempts",
            "Need permissions for shared folder",
            "Cannot login to internal portal",
            "Two-factor authentication not working",
        ],
        "email": [
            "Outlook not syncing emails",
            "Teams meeting link not working",
            "Calendar invite not showing",
            "Cannot send attachments larger than 10MB",
            "Email signature disappeared",
        ],
    }

    bodies = {
        "network": [
            "Since the latest system update, my VPN drops every 30 minutes. This is blocking all remote work.",
            "The internet randomly disconnects several times per day causing workflow disruption.",
            "The firewall is blocking access to our analytics dashboard. Need it whitelisted urgently.",
        ],
        "hardware": [
            "My laptop screen goes black and flickers when on battery. Need this resolved asap.",
            "Several keys on my keyboard stopped working. Cannot type properly.",
            "The external monitor stopped being detected after a Windows update.",
        ],
        "software": [
            "The application crashes immediately after the loading screen with no error message.",
            "The installer fails with error code 0x80070005. Cannot complete the install.",
            "License expired and I cannot open any files. This is critical for today's deadline.",
        ],
        "access": [
            "I forgot my password and cannot login. Please reset my account.",
            "Account locked after 3 failed attempts. Urgent, I have a meeting in 10 minutes.",
            "I need read access to the Q3 reports folder for the board presentation.",
        ],
        "email": [
            "Outlook stopped syncing yesterday afternoon. I am missing important emails.",
            "The Teams link for my client call tomorrow is not working. Please help.",
            "My calendar shows overlapping meetings that I already declined.",
        ],
    }

    rows = []
    categories = list(category_subjects.keys())
    cat_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    for _ in range(n_samples):
        cat = rng.choice(categories, p=cat_weights)
        subject = rng.choice(category_subjects[cat])
        body = rng.choice(bodies[cat])
        priority = rng.choice(["low", "medium", "high", "critical"], p=[0.15, 0.45, 0.3, 0.1])
        created_hour = int(rng.integers(0, 24))
        created_dow = int(rng.integers(0, 7))
        org_size = int(rng.integers(50, 5000))
        lag_resolution = float(rng.exponential(8.0))
        rolling_breach = float(rng.beta(2, 8))

        # SLA breach: driven by priority and urgency
        priority_risk = {"low": 0.05, "medium": 0.15, "high": 0.35, "critical": 0.65}[priority]
        hour_risk = 0.1 if created_hour >= 17 or created_hour < 8 else 0.0
        sla_breach = int(rng.random() < priority_risk + hour_risk)

        # Resolution hours: priority-driven with noise
        base_hours = {"low": 24.0, "medium": 8.0, "high": 4.0, "critical": 1.5}[priority]
        resolution_hours = float(max(0.5, rng.normal(base_hours, base_hours * 0.3)))

        rows.append({
            "subject": subject,
            "body": body,
            "priority": priority,
            "created_hour": created_hour,
            "created_dow": created_dow,
            "org_size": org_size,
            "lag_resolution_hours": lag_resolution,
            "rolling_breach_rate": rolling_breach,
            "category": cat,
            "sla_breach": sla_breach,
            "resolution_hours": resolution_hours,
        })

    return pd.DataFrame(rows)


def train_models(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Train category classifier, SLA breach predictor, and resolution estimator.

    Args:
        df: Training DataFrame. Generates synthetic data if None.

    Returns:
        Dictionary of cross-validation metrics for all three models.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if df is None:
        logger.info("Generating synthetic training data")
        df = generate_synthetic_data(n_samples=3000)

    X = df.drop(columns=["category", "sla_breach", "resolution_hours"])
    y_cat = df["category"]
    y_breach = df["sla_breach"]
    y_hours = df["resolution_hours"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Training category classifier")
    cat_pipe = _make_category_pipeline()
    cat_scores = cross_val_score(cat_pipe, X, y_cat, cv=cv, scoring="accuracy")
    cat_pipe.fit(X, y_cat)
    joblib.dump(cat_pipe, MODEL_DIR / "category_model.joblib")

    logger.info("Training SLA breach predictor")
    breach_pipe = _make_breach_pipeline()
    breach_scores = cross_val_score(breach_pipe, X, y_breach, cv=cv, scoring="roc_auc")
    breach_pipe.fit(X, y_breach)
    joblib.dump(breach_pipe, MODEL_DIR / "breach_model.joblib")

    logger.info("Training resolution hours estimator")
    res_pipe = _make_resolution_pipeline()
    res_scores = cross_val_score(res_pipe, X, y_hours, cv=5, scoring="r2")
    res_pipe.fit(X, y_hours)
    joblib.dump(res_pipe, MODEL_DIR / "resolution_model.joblib")

    metrics = {
        "category_accuracy_mean": float(cat_scores.mean()),
        "category_accuracy_std": float(cat_scores.std()),
        "breach_auc_mean": float(breach_scores.mean()),
        "breach_auc_std": float(breach_scores.std()),
        "resolution_r2_mean": float(res_scores.mean()),
        "resolution_r2_std": float(res_scores.std()),
        "n_train": len(df),
        "model_version": MODEL_VERSION,
    }

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Training complete: cat_acc=%.3f breach_auc=%.3f res_r2=%.3f",
        metrics["category_accuracy_mean"],
        metrics["breach_auc_mean"],
        metrics["resolution_r2_mean"],
    )
    return metrics


def load_models() -> tuple[Pipeline, Pipeline, Pipeline]:
    """Load trained models from disk, training them if not present.

    Returns:
        Tuple of (category_pipe, breach_pipe, resolution_pipe).
    """
    cat_path = MODEL_DIR / "category_model.joblib"
    breach_path = MODEL_DIR / "breach_model.joblib"
    res_path = MODEL_DIR / "resolution_model.joblib"

    if not cat_path.exists() or not breach_path.exists() or not res_path.exists():
        logger.info("Models not found, training now")
        train_models()

    cat_pipe = joblib.load(cat_path)
    breach_pipe = joblib.load(breach_path)
    res_pipe = joblib.load(res_path)
    return cat_pipe, breach_pipe, res_pipe


def predict(
    ticket_df: pd.DataFrame,
    cat_pipe: Pipeline,
    breach_pipe: Pipeline,
    res_pipe: Pipeline,
) -> list[dict[str, Any]]:
    """Run all three models on a batch of tickets.

    Args:
        ticket_df: DataFrame with raw ticket columns.
        cat_pipe: Trained category classifier pipeline.
        breach_pipe: Trained SLA breach predictor pipeline.
        res_pipe: Trained resolution hours estimator pipeline.

    Returns:
        List of prediction dicts per ticket.
    """
    cat_preds = cat_pipe.predict(ticket_df)
    cat_probs = cat_pipe.predict_proba(ticket_df).max(axis=1)
    breach_probs = breach_pipe.predict_proba(ticket_df)[:, 1]
    res_preds = res_pipe.predict(ticket_df)

    results = []
    for i in range(len(ticket_df)):
        results.append({
            "predicted_category": cat_preds[i],
            "confidence": float(cat_probs[i]),
            "sla_breach_prob": float(breach_probs[i]),
            "resolution_hours_pred": float(max(0.0, res_preds[i])),
        })
    return results
