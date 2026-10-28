"""ML model training, loading and inference for News-Guard."""

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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import TextFeatureExtractor, TfidfHeadlineTransformer

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
MODEL_VERSION = "1.0.0"


def _build_pipeline() -> Pipeline:
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft",
        weights=[2, 1],
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ensemble),
    ])


def generate_synthetic_data(n_samples: int = 2000) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic training data for demonstration."""
    rng = np.random.default_rng(42)
    rows = []
    labels = []

    fake_headlines = [
        "You Won't Believe What Happened Next!", "SHOCKING: Secret Government Plot Exposed!",
        "Scientists Baffled by This One Weird Trick", "Breaking: Explosive Revelation!!!",
        "They Don't Want You to Know This", "This Will Blow Your Mind",
        "Government Hides the Truth About", "BOMBSHELL: What They're Not Telling You",
    ]
    real_headlines = [
        "Reuters: Central bank holds interest rates steady",
        "Study finds link between diet and cardiovascular health",
        "City council votes on infrastructure spending bill",
        "Technology company reports quarterly earnings",
        "Scientists publish findings on climate change effects",
        "Health officials recommend updated vaccination guidelines",
    ]
    unreliable = ["infowars.com", "naturalnews.com", "beforeitsnews.com"]
    reliable = ["reuters.com", "apnews.com", "bbc.com", "nytimes.com"]

    for i in range(n_samples):
        label = rng.integers(0, 2)
        if label == 1:
            headline = fake_headlines[i % len(fake_headlines)]
            source = unreliable[i % len(unreliable)]
            body = "Experts claim " + " ".join(["word"] * rng.integers(20, 100))
            extra_excl = "!!!" * rng.integers(0, 5)
            headline = headline + extra_excl
        else:
            headline = real_headlines[i % len(real_headlines)]
            source = reliable[i % len(reliable)]
            body = "According to sources, " + " ".join(["data"] * rng.integers(50, 200))

        rows.append({
            "headline": headline,
            "body": body,
            "source_domain": source,
            "publication_hour": int(rng.integers(0, 24)),
            "share_count": int(rng.integers(0, 10000)),
        })
        labels.append(int(label))

    return pd.DataFrame(rows), np.array(labels)


class NewsGuardModel:
    """Ensemble model for fake news classification."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._text_extractor = TextFeatureExtractor()
        self._tfidf = TfidfHeadlineTransformer(max_features=300)
        self._scaler = StandardScaler()
        self._inner_pipeline: Pipeline | None = None
        self.version = MODEL_VERSION

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        hand_crafted = self._text_extractor.transform(df)
        tfidf_feats = self._tfidf.transform(df)
        return np.hstack([hand_crafted, tfidf_feats])

    def train(self, df: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
        self._text_extractor.fit(df)
        self._tfidf.fit(df)
        X = self._prepare_X(df)

        self._inner_pipeline = _build_pipeline()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self._inner_pipeline, X, y, cv=cv, scoring="roc_auc")
        self._inner_pipeline.fit(X, y)
        self._scaler = self._inner_pipeline.named_steps["scaler"]

        metrics = {
            "auc_mean": round(float(scores.mean()), 4),
            "auc_std": round(float(scores.std()), 4),
            "n_features": X.shape[1],
            "n_samples": len(y),
            "model_version": MODEL_VERSION,
        }
        joblib.dump(self, str(MODEL_PATH))
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        logger.info("Model trained. AUC=%.4f ± %.4f", scores.mean(), scores.std())
        return metrics

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        if self._inner_pipeline is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = self._prepare_X(df)
        proba = self._inner_pipeline.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))
        label = "FAKE" if label_idx == 1 else "REAL"
        confidence = float(proba[label_idx])
        credibility = round(1.0 - float(proba[1]), 4)
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "credibility_score": credibility,
            "fake_probability": round(float(proba[1]), 4),
            "real_probability": round(float(proba[0]), 4),
        }

    def get_reference_features(self) -> list[float]:
        """Return reference distribution (fake proba) for drift detection."""
        df, _ = generate_synthetic_data(200)
        X = self._prepare_X(df)
        return [float(p) for p in self._inner_pipeline.predict_proba(X)[:, 1]]

    @classmethod
    def load(cls) -> "NewsGuardModel":
        if not MODEL_PATH.exists():
            logger.info("No model found; training on synthetic data.")
            inst = cls()
            df, y = generate_synthetic_data()
            inst.train(df, y)
            return inst
        return joblib.load(str(MODEL_PATH))
