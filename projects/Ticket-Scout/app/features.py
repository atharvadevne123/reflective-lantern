"""Feature engineering pipeline for support ticket data."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PRIORITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}

URGENCY_KEYWORDS = [
    "urgent", "critical", "asap", "immediately", "broken", "down",
    "cannot", "failed", "error", "outage", "blocked", "emergency",
]

CATEGORY_KEYWORDS = {
    "network": ["vpn", "network", "wifi", "internet", "connection", "firewall"],
    "hardware": ["laptop", "screen", "keyboard", "mouse", "monitor", "printer"],
    "software": ["install", "software", "crash", "update", "license", "application"],
    "access": ["password", "login", "access", "permission", "account", "locked"],
    "email": ["email", "outlook", "mailbox", "calendar", "teams", "meeting"],
}


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _count_urgency(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)


def _category_keyword_scores(text: str) -> dict[str, float]:
    text_lower = text.lower()
    return {
        f"kw_{cat}": sum(1 for kw in kws if kw in text_lower)
        for cat, kws in CATEGORY_KEYWORDS.items()
    }


def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric features from raw ticket DataFrame.

    Args:
        df: Raw tickets with columns subject, body, priority, created_hour,
            created_dow, org_size.

    Returns:
        DataFrame of numeric features ready for model input.
    """
    feats = pd.DataFrame(index=df.index)

    combined = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(_clean_text)

    feats["text_length"] = combined.str.len()
    feats["word_count"] = combined.str.split().str.len()
    feats["urgency_score"] = combined.apply(_count_urgency)
    feats["subject_length"] = df["subject"].fillna("").str.len()
    feats["has_question"] = df["body"].fillna("").str.contains(r"\?").astype(int)

    kw_df = combined.apply(_category_keyword_scores).apply(pd.Series)
    feats = pd.concat([feats, kw_df], axis=1)

    feats["priority_num"] = (
        df["priority"].fillna("medium").str.lower().map(PRIORITY_MAP).fillna(1).astype(int)
    )
    feats["created_hour"] = df.get("created_hour", pd.Series(12, index=df.index))
    feats["created_dow"] = df.get("created_dow", pd.Series(0, index=df.index))
    feats["org_size_log"] = np.log1p(df.get("org_size", pd.Series(500, index=df.index)))

    # Interaction features
    feats["urgency_x_priority"] = feats["urgency_score"] * feats["priority_num"]
    feats["text_x_urgency"] = feats["text_length"] * feats["urgency_score"]

    # Rolling/lag placeholder (filled with mean when single row)
    feats["lag_resolution_hours"] = df.get(
        "lag_resolution_hours", pd.Series(8.0, index=df.index)
    )
    feats["rolling_breach_rate"] = df.get(
        "rolling_breach_rate", pd.Series(0.1, index=df.index)
    )

    return feats.fillna(0)


class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    """Converts raw ticket dicts into a numeric feature matrix."""

    def __init__(self, max_tfidf_features: int = 200) -> None:
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_ = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.scaler_ = StandardScaler()
        self._fitted = False

    def _to_df(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def fit(self, X: Any, y: Any = None) -> TextFeatureTransformer:
        df = self._to_df(X)
        struct = extract_structured_features(df)
        combined = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(_clean_text)
        self.tfidf_.fit(combined)
        tfidf_matrix = self.tfidf_.transform(combined).toarray()
        full = np.hstack([struct.values, tfidf_matrix])
        self.scaler_.fit(full)
        self._fitted = True
        logger.info("TextFeatureTransformer fitted on %d samples, %d features", len(df), full.shape[1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        df = self._to_df(X)
        struct = extract_structured_features(df)
        combined = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(_clean_text)
        tfidf_matrix = self.tfidf_.transform(combined).toarray()
        full = np.hstack([struct.values, tfidf_matrix])
        return self.scaler_.transform(full)

    def get_feature_names(self) -> list[str]:
        struct_cols = [
            "text_length", "word_count", "urgency_score", "subject_length",
            "has_question", "kw_network", "kw_hardware", "kw_software",
            "kw_access", "kw_email", "priority_num", "created_hour",
            "created_dow", "org_size_log", "urgency_x_priority",
            "text_x_urgency", "lag_resolution_hours", "rolling_breach_rate",
        ]
        tfidf_cols = [f"tfidf_{t}" for t in self.tfidf_.get_feature_names_out()]
        return struct_cols + tfidf_cols
