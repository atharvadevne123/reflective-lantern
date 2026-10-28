"""Feature engineering pipeline for news article misinformation detection."""

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

CLICKBAIT_PATTERNS = [
    r"\bYou Won't Believe\b",
    r"\bShocking\b",
    r"\bBreaking\b",
    r"\bGoes Viral\b",
    r"\bThis Is Why\b",
    r"\bWill Blow Your Mind\b",
    r"\b\d+ Reasons\b",
    r"\bExperts (Say|Warn|Reveal)\b",
    r"\bSecret(s)?\b",
    r"!{2,}",
]

KNOWN_UNRELIABLE_DOMAINS = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "yournewswire.com", "theonion.com", "babylonbee.com",
    "clickhole.com", "worldnewsdailyreport.com",
}

KNOWN_RELIABLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "npr.org", "pbs.org", "economist.com",
}


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts hand-crafted linguistic features from article text."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TextFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in X.iterrows():
            headline = str(row.get("headline", ""))
            body = str(row.get("body", ""))
            source = str(row.get("source_domain", ""))
            rows.append(self._extract(headline, body, source))
        return np.array(rows, dtype=np.float32)

    def _extract(self, headline: str, body: str, source: str) -> list[float]:
        full_text = f"{headline} {body}"
        words = full_text.split()
        sentences = re.split(r"[.!?]+", full_text)

        clickbait_score = sum(
            1 for p in CLICKBAIT_PATTERNS if re.search(p, headline, re.IGNORECASE)
        )
        caps_ratio = sum(1 for c in headline if c.isupper()) / max(len(headline), 1)
        exclamation_count = headline.count("!") + body.count("!")
        question_count = headline.count("?") + body.count("?")
        avg_sentence_len = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0.0
        vocab_richness = len(set(w.lower() for w in words)) / max(len(words), 1)
        word_count = len(words)
        source_unreliable = float(source in KNOWN_UNRELIABLE_DOMAINS)
        source_reliable = float(source in KNOWN_RELIABLE_DOMAINS)
        num_quotes = full_text.count('"') // 2
        hedge_words = sum(1 for w in ["allegedly", "reportedly", "claimed", "sources say"] if w in full_text.lower())
        sensational_words = sum(1 for w in ["bombshell", "explosive", "devastating", "unprecedented"] if w in full_text.lower())
        url_mentions = len(re.findall(r"http[s]?://", full_text))
        return [
            float(clickbait_score),
            float(caps_ratio),
            float(exclamation_count),
            float(question_count),
            float(avg_sentence_len),
            float(vocab_richness),
            float(word_count),
            source_unreliable,
            source_reliable,
            float(num_quotes),
            float(hedge_words),
            float(sensational_words),
            float(url_mentions),
        ]

    def get_feature_names_out(self) -> list[str]:
        return [
            "clickbait_score", "caps_ratio", "exclamation_count", "question_count",
            "avg_sentence_len", "vocab_richness", "word_count",
            "source_unreliable", "source_reliable", "num_quotes",
            "hedge_words", "sensational_words", "url_mentions",
        ]


class TfidfHeadlineTransformer(BaseEstimator, TransformerMixin):
    """TF-IDF on headlines with n-gram features."""

    def __init__(self, max_features: int = 500, ngram_range: tuple = (1, 2)) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vec: TfidfVectorizer | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TfidfHeadlineTransformer":
        headlines = X["headline"].fillna("").astype(str).tolist()
        self._vec = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            stop_words="english",
        )
        self._vec.fit(headlines)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        headlines = X["headline"].fillna("").astype(str).tolist()
        return self._vec.transform(headlines).toarray()


def compute_feature_hash(features: dict) -> str:
    """Deterministic hash of input feature dict for deduplication."""
    canonical = str(sorted(features.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_feature_dataframe(
    headline: str,
    body: str,
    source_domain: str,
    publication_hour: int = 12,
    share_count: int = 0,
) -> pd.DataFrame:
    """Build a single-row DataFrame from raw article inputs."""
    return pd.DataFrame([{
        "headline": headline,
        "body": body,
        "source_domain": source_domain,
        "publication_hour": publication_hour,
        "share_count": share_count,
    }])
