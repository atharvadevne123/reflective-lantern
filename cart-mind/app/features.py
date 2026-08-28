"""Feature engineering for Cart-Mind.

Five sklearn transformers turn 16 raw user, item, and interaction columns into 29
features. They are transformers rather than a preprocessing function so they can
be fitted inside cross-validation folds: :class:`LagRollingFeatures` learns
normalisation constants from the training data, and computing those over the full
dataset would leak test-fold information into the reported AUC.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature column names expected in raw input
USER_COLS = [
    "user_age",
    "purchase_count",
    "avg_order_value",
    "days_since_last_purchase",
    "days_since_registration",
    "session_count_7d",
    "cart_abandon_rate",
]
ITEM_COLS = [
    "item_price",
    "item_avg_rating",
    "item_review_count",
    "item_inventory_level",
    "item_discount_pct",
]
INTERACTION_COLS = [
    "view_count",
    "click_count",
    "wishlist_flag",
    "same_category_purchases",
]


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Cross-column ratios capturing relative rather than absolute signals.

    A £200 item means something different to a shopper whose average order is £50
    than to one who routinely spends £500, so the ratios carry signal the raw
    columns cannot. All divisions are epsilon-guarded against zero denominators.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> RatioFeatures:
        """No-op fit; returns self for pipeline compatibility."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute epsilon-guarded ratio features from price, rating, and engagement columns."""
        X = X.copy()
        eps = 1e-6
        X["price_per_rating"] = X["item_price"] / (X["item_avg_rating"] + eps)
        X["value_score"] = X["item_avg_rating"] * np.log1p(X["item_review_count"])
        X["engagement_rate"] = X["click_count"] / (X["view_count"] + eps)
        X["purchase_intensity"] = X["purchase_count"] / (X["days_since_registration"] + eps)
        X["spend_per_purchase"] = X["avg_order_value"] / (X["purchase_count"] + eps)
        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """User-item affinity, recency decay, and price sensitivity.

    ``recency_weight`` decays exponentially with a 30-day scale, encoding that a
    lapsed shopper's history should count for less without discarding it outright.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> InteractionFeatures:
        """No-op fit; returns self for pipeline compatibility."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add affinity score, recency decay weight, and price sensitivity columns."""
        X = X.copy()
        X["affinity_score"] = (
            X["same_category_purchases"] * 2.0 + X["wishlist_flag"] * 3.0 + X["click_count"] * 0.5
        ) / 10.0
        X["recency_weight"] = np.exp(-X["days_since_last_purchase"] / 30.0)
        X["price_sensitivity"] = np.where(
            X["avg_order_value"] > 0,
            X["item_price"] / X["avg_order_value"],
            1.0,
        )
        return X


class LagRollingFeatures(BaseEstimator, TransformerMixin):
    """Purchase-velocity features normalised against training-set means.

    The means are learned in ``fit`` and reused in ``transform``, so a fold's
    normalisation never sees data from outside that fold.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> LagRollingFeatures:
        """Learn training-set means for session count and order value normalisation."""
        self.session_mean_ = float(X["session_count_7d"].mean())
        self.order_mean_ = float(X["avg_order_value"].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned normalisation and derive purchase velocity."""
        X = X.copy()
        X["session_norm"] = X["session_count_7d"] / (self.session_mean_ + 1e-6)
        X["order_value_norm"] = X["avg_order_value"] / (self.order_mean_ + 1e-6)
        X["purchase_velocity"] = X["purchase_count"] / (
            np.maximum(X["days_since_registration"], 1)
        )
        return X


class DiscountEncoder(BaseEstimator, TransformerMixin):
    """Discount buckets and a low-stock scarcity flag.

    Discount is bucketed rather than left continuous because its effect on
    conversion is closer to stepwise than linear — shoppers respond to "half
    price" as a category, not to each percentage point equally.
    """

    DISCOUNT_BINS = [0, 5, 15, 30, 50, 100]
    DISCOUNT_LABELS = [0, 1, 2, 3, 4]

    def fit(self, X: pd.DataFrame, y: Any = None) -> DiscountEncoder:
        """No-op fit; returns self for pipeline compatibility."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Bucket discount percentage and add inventory scarcity flag."""
        X = X.copy()
        X["discount_bucket"] = pd.cut(
            X["item_discount_pct"],
            bins=self.DISCOUNT_BINS,
            labels=self.DISCOUNT_LABELS,
            include_lowest=True,
        ).astype(float)
        X["inventory_pressure"] = np.where(X["item_inventory_level"] < 10, 1.0, 0.0)
        return X


def build_feature_pipeline() -> Pipeline:
    """Build the five-stage feature pipeline.

    Returns:
        An unfitted pipeline: ratios, interactions, lag/rolling, discount
        encoding, then standard scaling.
    """
    return Pipeline(
        [
            ("ratios", RatioFeatures()),
            ("interactions", InteractionFeatures()),
            ("lag_rolling", LagRollingFeatures()),
            ("discount_encoding", DiscountEncoder()),
            ("scaler", StandardScaler()),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return final feature column names after pipeline transformation."""
    base = (
        USER_COLS
        + ITEM_COLS
        + INTERACTION_COLS
        + [
            "price_per_rating",
            "value_score",
            "engagement_rate",
            "purchase_intensity",
            "spend_per_purchase",
            "affinity_score",
            "recency_weight",
            "price_sensitivity",
            "session_norm",
            "order_value_norm",
            "purchase_velocity",
            "discount_bucket",
            "inventory_pressure",
        ]
    )
    return base


def make_sample_dataframe(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic interaction dataset.

    This repository ships no proprietary retail data, so training and tests run
    on generated rows. Pair with :func:`make_purchase_labels` for labels that
    carry signal; random labels produce a near-chance model.

    Args:
        n: Number of rows to generate.
        seed: Seed for reproducibility.

    Returns:
        Frame with all 16 raw feature columns.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "user_age": rng.integers(18, 70, n),
            "purchase_count": rng.integers(0, 200, n),
            "avg_order_value": rng.uniform(10, 500, n),
            "days_since_last_purchase": rng.integers(0, 365, n),
            "days_since_registration": rng.integers(1, 1825, n),
            "session_count_7d": rng.integers(0, 30, n),
            "cart_abandon_rate": rng.uniform(0, 1, n),
            "item_price": rng.uniform(5, 1000, n),
            "item_avg_rating": rng.uniform(1, 5, n),
            "item_review_count": rng.integers(0, 10000, n),
            "item_inventory_level": rng.integers(0, 500, n),
            "item_discount_pct": rng.uniform(0, 80, n),
            "view_count": rng.integers(0, 50, n),
            "click_count": rng.integers(0, 20, n),
            "wishlist_flag": rng.integers(0, 2, n),
            "same_category_purchases": rng.integers(0, 30, n),
        }
    )


def make_purchase_labels(df: pd.DataFrame, seed: int = 42, noise: float = 0.35) -> pd.Series:
    """Generate signal-bearing purchase labels from a feature frame.

    Builds a latent purchase-propensity score from behavioural drivers that
    genuinely predict conversion — engagement, wishlist intent, category
    affinity, recency, discount depth, and price friction — then samples binary
    outcomes from a logistic transform of that score. The ``noise`` term keeps
    the problem non-separable so cross-validated AUC lands in a realistic band
    rather than saturating at 1.0.

    Args:
        df: Feature frame produced by :func:`make_sample_dataframe`.
        seed: Seed for the label-sampling generator.
        noise: Standard deviation of Gaussian noise added to the latent score.

    Returns:
        Binary purchase labels aligned to ``df``'s index.
    """
    rng = np.random.default_rng(seed)

    def _z(col: str) -> np.ndarray:
        """Z-score normalise column *col* from the outer DataFrame *df*."""
        v = df[col].to_numpy(dtype=float)
        return (v - v.mean()) / (v.std() + 1e-6)

    latent = (
        0.55 * _z("click_count")
        + 0.45 * df["wishlist_flag"].to_numpy(dtype=float)
        + 0.40 * _z("same_category_purchases")
        + 0.30 * _z("item_avg_rating")
        + 0.25 * _z("item_discount_pct")
        + 0.20 * _z("session_count_7d")
        - 0.35 * _z("item_price")
        - 0.30 * _z("days_since_last_purchase")
        - 0.25 * _z("cart_abandon_rate")
    )
    latent = latent + rng.normal(0, noise * latent.std(), len(df))
    prob = 1.0 / (1.0 + np.exp(-latent))
    return pd.Series((rng.random(len(df)) < prob).astype(int), index=df.index, name="purchased")


def feature_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix for the numeric columns of *df*.

    Useful during feature selection and debugging to spot highly correlated
    pairs that may degrade tree-based models via redundancy.

    Args:
        df: DataFrame with at least one numeric column.

    Returns:
        Correlation matrix as a DataFrame (NaN entries replaced with 0.0).
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr().fillna(0.0)


def drop_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Return a copy of *df* with near-zero-variance columns removed.

    A feature whose standard deviation is below *threshold* carries almost no
    discriminative signal and can destabilise normalisation. This is a quick
    pre-flight check before fitting the feature pipeline.

    Args:
        df: Input feature frame.
        threshold: Columns with std < threshold are dropped.

    Returns:
        Filtered DataFrame (copy).
    """
    numeric = df.select_dtypes(include="number")
    low_var = [col for col in numeric.columns if numeric[col].std() < threshold]
    if low_var:
        logger.debug("drop_low_variance_features: dropping %d columns: %s", len(low_var), low_var)
    return df.drop(columns=low_var)
