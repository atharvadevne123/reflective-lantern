"""NLP feature engineering pipeline for viral content prediction."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

logger = logging.getLogger(__name__)

_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF]+",
    flags=re.UNICODE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_EXCLAIM_RE = re.compile(r"!")
_QUESTION_RE = re.compile(r"\?")


def count_emojis(text: str) -> int:
    """Return the number of emoji characters in *text*."""
    return len(_EMOJI_RE.findall(text))


def count_mentions(text: str) -> int:
    """Return the number of @mention tokens in *text*."""
    return len(_MENTION_RE.findall(text))


def count_hashtags(text: str) -> int:
    """Return the number of #hashtag tokens in *text*."""
    return len(_HASHTAG_RE.findall(text))


def has_url(text: str) -> int:
    """Return 1 if *text* contains at least one URL, else 0."""
    return int(bool(_URL_RE.search(text)))


def exclamation_density(text: str) -> float:
    """Fraction of characters that are exclamation marks."""
    if not text:
        return 0.0
    return len(_EXCLAIM_RE.findall(text)) / len(text)


def question_density(text: str) -> float:
    """Fraction of characters that are question marks."""
    if not text:
        return 0.0
    return len(_QUESTION_RE.findall(text)) / len(text)


def capitalisation_ratio(text: str) -> float:
    """Ratio of uppercase letters to total alpha characters."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def avg_word_length(text: str) -> float:
    """Mean length of tokens after stripping punctuation."""
    words = [w.strip(".,!?;:\"'") for w in text.split() if w.strip(".,!?;:\"'")]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def encode_cyclical(value: int, period: int) -> tuple[float, float]:
    """Encode an integer with cyclical sine/cosine encoding."""
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def follower_bucket(followers: int) -> int:
    """Map follower count to a log-scale bucket 0-6."""
    if followers < 100:
        return 0
    if followers < 1_000:
        return 1
    if followers < 10_000:
        return 2
    if followers < 100_000:
        return 3
    if followers < 1_000_000:
        return 4
    if followers < 10_000_000:
        return 5
    return 6


class TextStatTransformer(BaseEstimator, TransformerMixin):
    """Extract hand-crafted text statistics from a list of dicts."""

    FEATURE_NAMES: list[str] = [
        "text_length",
        "word_count",
        "hashtag_count",
        "mention_count",
        "emoji_count",
        "has_url",
        "exclamation_density",
        "question_density",
        "capitalisation_ratio",
        "avg_word_length",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "follower_bucket",
        "has_media",
        "has_link",
    ]

    def fit(self, X: list[dict[str, Any]], y: Any = None) -> "TextStatTransformer":
        return self

    def transform(self, X: list[dict[str, Any]]) -> np.ndarray:
        """Build a float matrix, one row per sample."""
        rows: list[list[float]] = []
        for item in X:
            text: str = item.get("text", "") or ""
            hour: int = int(item.get("hour_of_day", 12))
            dow: int = int(item.get("day_of_week", 0))
            followers: int = int(item.get("author_followers", 0))
            hour_sin, hour_cos = encode_cyclical(hour, 24)
            dow_sin, dow_cos = encode_cyclical(dow, 7)
            row = [
                len(text),
                len(text.split()),
                count_hashtags(text),
                count_mentions(text),
                count_emojis(text),
                has_url(text),
                exclamation_density(text),
                question_density(text),
                capitalisation_ratio(text),
                avg_word_length(text),
                hour_sin,
                hour_cos,
                dow_sin,
                dow_cos,
                follower_bucket(followers),
                int(item.get("has_media", False)),
                int(item.get("has_link", False)),
            ]
            rows.append(row)
        result = np.array(rows, dtype=np.float32)
        logger.debug("TextStatTransformer: produced shape %s", result.shape)
        return result

    def get_feature_names_out(self) -> list[str]:
        return self.FEATURE_NAMES


class TextExtractor(BaseEstimator, TransformerMixin):
    """Pull the raw text string out of a dict for TF-IDF."""

    def fit(self, X: list[dict[str, Any]], y: Any = None) -> "TextExtractor":
        return self

    def transform(self, X: list[dict[str, Any]]) -> list[str]:
        return [item.get("text", "") or "" for item in X]


def build_tfidf_pipeline(max_features: int = 500) -> Pipeline:
    """Return a pipeline that extracts text and vectorises it."""
    return Pipeline([
        ("extract", TextExtractor()),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
    ])


def build_feature_pipeline(tfidf_features: int = 200) -> FeatureUnion:
    """Combine hand-crafted stats with TF-IDF into a single transformer.

    The resulting feature union produces a dense + sparse matrix that
    scikit-learn's ensemble models consume directly.
    """
    return FeatureUnion([
        ("stats", TextStatTransformer()),
        ("tfidf", build_tfidf_pipeline(max_features=tfidf_features)),
    ])
