"""Feature engineering pipeline for sports match prediction.

Transforms raw match statistics into a 30+ dimensional feature vector
via a 7-stage sklearn-compatible Pipeline:
    FormEncoder → LagRollingTransformer → HeadToHeadTransformer
    → RatioFeatureTransformer → FatigueTransformer
    → DropCategoricalTransformer → StandardScaler
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FormEncoder(BaseEstimator, TransformerMixin):
    """Encode recent form as win-rate, unbeaten-rate, and momentum.

    Args:
        BaseEstimator: sklearn base class for get/set_params.
        TransformerMixin: sklearn mixin for fit_transform shorthand.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "FormEncoder":
        """No-op fit (stateless transformer).

        Args:
            X: Input DataFrame with form columns.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute form ratio features for home and away teams.

        Args:
            X: DataFrame with *_wins_last5, *_draws_last5, *_losses_last5 columns.

        Returns:
            DataFrame augmented with *_win_rate, *_unbeaten_rate, *_momentum columns.
        """
        df = X.copy()
        for prefix in ("home", "away"):
            wins = df.get(f"{prefix}_wins_last5", pd.Series(0, index=df.index))
            draws = df.get(f"{prefix}_draws_last5", pd.Series(0, index=df.index))
            losses = df.get(f"{prefix}_losses_last5", pd.Series(0, index=df.index))
            total = (wins + draws + losses).replace(0, 1)
            df[f"{prefix}_win_rate"] = wins / total
            df[f"{prefix}_unbeaten_rate"] = (wins + draws) / total
            df[f"{prefix}_momentum"] = (wins * 3 + draws) / (total * 3)
        return df


class LagRollingTransformer(BaseEstimator, TransformerMixin):
    """Add lag and rolling aggregate features for goals and performance.

    Args:
        BaseEstimator: sklearn base class.
        TransformerMixin: sklearn mixin.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "LagRollingTransformer":
        """No-op fit.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive goal differential and attack/defense strength features.

        Args:
            X: DataFrame with *_goals_avg and *_goals_conceded_avg columns.

        Returns:
            DataFrame augmented with derived strength features.
        """
        df = X.copy()
        for prefix in ("home", "away"):
            goals = df.get(f"{prefix}_goals_avg", pd.Series(1.0, index=df.index))
            conceded = df.get(f"{prefix}_goals_conceded_avg", pd.Series(1.0, index=df.index))
            df[f"{prefix}_goal_diff_avg"] = goals - conceded
            df[f"{prefix}_attack_strength"] = goals / (goals.mean() + 1e-9)
            df[f"{prefix}_defense_strength"] = 1.0 / (conceded + 1e-9)
        return df


class HeadToHeadTransformer(BaseEstimator, TransformerMixin):
    """Encode head-to-head historical statistics.

    Args:
        BaseEstimator: sklearn base class.
        TransformerMixin: sklearn mixin.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "HeadToHeadTransformer":
        """No-op fit.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute H2H win-rates and home-advantage delta.

        Args:
            X: DataFrame with h2h_home_wins, h2h_away_wins, h2h_draws columns.

        Returns:
            DataFrame augmented with H2H ratio features.
        """
        df = X.copy()
        h2h_home = df.get("h2h_home_wins", pd.Series(1, index=df.index))
        h2h_away = df.get("h2h_away_wins", pd.Series(1, index=df.index))
        h2h_draws = df.get("h2h_draws", pd.Series(0, index=df.index))
        total = (h2h_home + h2h_away + h2h_draws).replace(0, 1)
        df["h2h_home_win_rate"] = h2h_home / total
        df["h2h_away_win_rate"] = h2h_away / total
        df["h2h_draw_rate"] = h2h_draws / total
        df["h2h_home_advantage"] = df["h2h_home_win_rate"] - df["h2h_away_win_rate"]
        return df


class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Compute relative-strength ratios between home and away teams.

    Args:
        BaseEstimator: sklearn base class.
        TransformerMixin: sklearn mixin.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "RatioFeatureTransformer":
        """No-op fit.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive Elo-based expected win probability and ranking differential.

        Args:
            X: DataFrame with home_ranking, away_ranking, home_elo, away_elo columns.

        Returns:
            DataFrame augmented with Elo diff, ratio, and expected win probability.
        """
        df = X.copy()
        home_rank = df.get("home_ranking", pd.Series(10.0, index=df.index))
        away_rank = df.get("away_ranking", pd.Series(10.0, index=df.index))
        home_elo = df.get("home_elo", pd.Series(1500.0, index=df.index))
        away_elo = df.get("away_elo", pd.Series(1500.0, index=df.index))
        df["ranking_diff"] = away_rank - home_rank
        df["elo_diff"] = home_elo - away_elo
        df["elo_ratio"] = home_elo / (away_elo + 1e-9)
        df["elo_win_prob"] = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
        return df


class FatigueTransformer(BaseEstimator, TransformerMixin):
    """Encode player/team fatigue from days since last match.

    Args:
        BaseEstimator: sklearn base class.
        TransformerMixin: sklearn mixin.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "FatigueTransformer":
        """No-op fit.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute exponential fatigue decay and rest-advantage features.

        Args:
            X: DataFrame with *_days_rest columns.

        Returns:
            DataFrame augmented with *_fatigue and *_rest_advantage columns.
        """
        df = X.copy()
        for prefix in ("home", "away"):
            days_rest = df.get(f"{prefix}_days_rest", pd.Series(7, index=df.index))
            df[f"{prefix}_fatigue"] = np.exp(-days_rest / 7.0)
            df[f"{prefix}_rest_advantage"] = np.clip(days_rest / 14.0, 0.0, 1.0)
        return df


class DropCategoricalTransformer(BaseEstimator, TransformerMixin):
    """Drop non-numeric columns that cannot be passed to the model.

    Args:
        BaseEstimator: sklearn base class.
        TransformerMixin: sklearn mixin.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "DropCategoricalTransformer":
        """Identify numeric columns to retain.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        self.numeric_cols_: list[str] = [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        logger.debug("DropCategoricalTransformer keeping %d columns", len(self.numeric_cols_))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Retain only numeric columns and cast to float32.

        Args:
            X: Input DataFrame.

        Returns:
            numpy ndarray of shape (n_samples, n_numeric_features).
        """
        return X[self.numeric_cols_].values.astype(np.float32)


FEATURE_COLUMNS: list[str] = [
    "home_wins_last5", "home_draws_last5", "home_losses_last5",
    "away_wins_last5", "away_draws_last5", "away_losses_last5",
    "home_goals_avg", "home_goals_conceded_avg",
    "away_goals_avg", "away_goals_conceded_avg",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "home_ranking", "away_ranking",
    "home_elo", "away_elo",
    "home_days_rest", "away_days_rest",
    "home_is_home_ground",
]


def make_feature_pipeline() -> Pipeline:
    """Build the 7-stage sport feature engineering pipeline.

    Returns:
        sklearn Pipeline ready for fit_transform.
    """
    return Pipeline([
        ("form", FormEncoder()),
        ("lag_rolling", LagRollingTransformer()),
        ("h2h", HeadToHeadTransformer()),
        ("ratios", RatioFeatureTransformer()),
        ("fatigue", FatigueTransformer()),
        ("drop_cat", DropCategoricalTransformer()),
        ("scaler", StandardScaler()),
    ])


def make_synthetic_dataset(n: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a signal-bearing synthetic sports match dataset.

    Labels are derived from a latent propensity score driven by Elo difference,
    form differential, and home-ground advantage with Gaussian noise. CV AUC
    above chance reflects recovery of this physical relationship, not real match data.

    Args:
        n: Number of match samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (feature DataFrame, label array) where labels are 0=away_win,
        1=draw, 2=home_win.
    """
    rng = np.random.default_rng(seed)

    home_elo = rng.normal(1500, 200, n).clip(800, 2200)
    away_elo = rng.normal(1500, 200, n).clip(800, 2200)
    home_days_rest = rng.integers(2, 14, n).astype(float)
    away_days_rest = rng.integers(2, 14, n).astype(float)

    df = pd.DataFrame({
        "home_wins_last5": rng.integers(0, 6, n),
        "home_draws_last5": rng.integers(0, 3, n),
        "home_losses_last5": rng.integers(0, 6, n),
        "away_wins_last5": rng.integers(0, 6, n),
        "away_draws_last5": rng.integers(0, 3, n),
        "away_losses_last5": rng.integers(0, 6, n),
        "home_goals_avg": rng.uniform(0.5, 3.0, n),
        "home_goals_conceded_avg": rng.uniform(0.5, 3.0, n),
        "away_goals_avg": rng.uniform(0.5, 3.0, n),
        "away_goals_conceded_avg": rng.uniform(0.5, 3.0, n),
        "h2h_home_wins": rng.integers(0, 10, n),
        "h2h_away_wins": rng.integers(0, 10, n),
        "h2h_draws": rng.integers(0, 5, n),
        "home_ranking": rng.integers(1, 100, n).astype(float),
        "away_ranking": rng.integers(1, 100, n).astype(float),
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_days_rest": home_days_rest,
        "away_days_rest": away_days_rest,
        "home_is_home_ground": rng.integers(0, 2, n).astype(float),
    })

    elo_diff = home_elo - away_elo
    form_home = (df["home_wins_last5"] * 3 + df["home_draws_last5"]) / 15.0
    form_away = (df["away_wins_last5"] * 3 + df["away_draws_last5"]) / 15.0
    home_advantage = df["home_is_home_ground"] * 0.15
    latent = (
        0.4 * elo_diff / 400
        + 0.3 * (form_home - form_away)
        + home_advantage
        + rng.normal(0, 0.4, n)
    )

    labels = np.where(latent > 0.3, 2, np.where(latent < -0.3, 0, 1))
    logger.debug("Synthetic dataset: n=%d, class distribution=%s", n, dict(zip(*np.unique(labels, return_counts=True))))
    return df, labels
