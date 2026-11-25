"""Feature engineering pipeline for IT helpdesk ticket classification."""
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

PRIORITY_KEYWORDS = {
    "critical": ["down", "outage", "critical", "urgent", "emergency", "prod", "production", "broken", "crash"],
    "high": ["slow", "error", "fail", "issue", "problem", "not working", "unable", "cannot", "blocked"],
    "medium": ["need", "help", "question", "how to", "request", "access", "setup", "configure"],
    "low": ["info", "fyi", "general", "feedback", "suggestion", "enhancement", "feature"],
}

CATEGORY_KEYWORDS = {
    "network": ["vpn", "wifi", "internet", "network", "connectivity", "latency", "ping"],
    "hardware": ["laptop", "computer", "monitor", "keyboard", "mouse", "printer", "device"],
    "software": ["install", "update", "upgrade", "application", "app", "software", "license"],
    "access": ["password", "login", "access", "permission", "account", "ldap", "sso", "mfa"],
    "email": ["email", "outlook", "calendar", "teams", "slack", "communication"],
    "security": ["virus", "malware", "phishing", "security", "breach", "suspicious", "hack"],
    "cloud": ["aws", "azure", "gcp", "cloud", "s3", "bucket", "vm", "kubernetes"],
    "data": ["database", "data", "query", "report", "analytics", "export", "import"],
}


class TextMetaExtractor(BaseEstimator, TransformerMixin):
    """Extracts numeric meta-features from ticket text fields."""

    def fit(self, X: Any, y: Any = None) -> "TextMetaExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in X.iterrows():
            subject = str(row.get("subject", "") or "").lower()
            desc = str(row.get("description", "") or "").lower()
            combined = subject + " " + desc

            subject_len = len(subject)
            desc_len = len(desc)
            word_count = len(combined.split())
            char_count = len(combined)
            avg_word_len = char_count / max(word_count, 1)
            sentence_count = max(len(re.split(r"[.!?]", desc)), 1)

            has_urgent = int(any(kw in combined for kw in PRIORITY_KEYWORDS["critical"]))
            has_high = int(any(kw in combined for kw in PRIORITY_KEYWORDS["high"]))
            has_medium = int(any(kw in combined for kw in PRIORITY_KEYWORDS["medium"]))
            has_low = int(any(kw in combined for kw in PRIORITY_KEYWORDS["low"]))

            category_counts = [
                sum(1 for kw in kws if kw in combined)
                for kws in CATEGORY_KEYWORDS.values()
            ]

            has_question = int("?" in combined)
            has_exclamation = int("!" in combined)
            caps_ratio = sum(1 for c in combined if c.isupper()) / max(len(combined), 1)
            digit_ratio = sum(1 for c in combined if c.isdigit()) / max(len(combined), 1)
            has_error_keyword = int(bool(re.search(r"error|exception|traceback|500|404", combined)))
            has_url = int(bool(re.search(r"https?://|www\.", combined)))

            features = [
                subject_len, desc_len, word_count, char_count, avg_word_len,
                sentence_count, has_urgent, has_high, has_medium, has_low,
                has_question, has_exclamation, caps_ratio, digit_ratio,
                has_error_keyword, has_url,
                *category_counts,
            ]
            rows.append(features)
        return np.array(rows, dtype=np.float32)


class CombinedTextExtractor(BaseEstimator, TransformerMixin):
    """Concatenates subject and description into a single string."""

    def fit(self, X: Any, y: Any = None) -> "CombinedTextExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> list[str]:
        return [
            (str(row.get("subject", "") or "") + " " + str(row.get("description", "") or "")).strip()
            for _, row in X.iterrows()
        ]


def build_feature_pipeline() -> FeatureUnion:
    """Build the full feature engineering pipeline."""
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )

    text_pipeline = Pipeline([
        ("extractor", CombinedTextExtractor()),
        ("tfidf", tfidf),
    ])

    meta_pipeline = Pipeline([
        ("meta", TextMetaExtractor()),
        ("scaler", StandardScaler()),
    ])

    return FeatureUnion([
        ("text", text_pipeline),
        ("meta", meta_pipeline),
    ])


def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate synthetic helpdesk ticket data for training."""
    rng = np.random.default_rng(42)
    priorities = ["critical", "high", "medium", "low"]
    categories = list(CATEGORY_KEYWORDS.keys())

    subjects_by_priority = {
        "critical": ["Production server down", "Critical outage affecting all users", "Emergency: system crash"],
        "high": ["Application error on login", "Cannot connect to VPN", "Email not working"],
        "medium": ["Request for software installation", "Need access to shared drive", "How to configure MFA"],
        "low": ["General inquiry about policies", "Feedback on new UI", "Request for feature enhancement"],
    }

    rows = []
    for _ in range(n_samples):
        priority = rng.choice(priorities, p=[0.1, 0.25, 0.4, 0.25])
        category = rng.choice(categories)
        subject_templates = subjects_by_priority[priority]
        subject = rng.choice(subject_templates)
        kws = CATEGORY_KEYWORDS[category]
        desc_kw = rng.choice(kws) if kws else "general issue"
        resolution_base = {"critical": 2, "high": 8, "medium": 24, "low": 72}[priority]
        resolution_hours = float(rng.lognormal(np.log(resolution_base), 0.5))
        rows.append({
            "subject": subject,
            "description": f"User reports {desc_kw} issue. " + " ".join(rng.choice(kws, min(3, len(kws)), replace=False)),
            "priority": priority,
            "category": category,
            "resolution_hours": resolution_hours,
        })

    return pd.DataFrame(rows)
