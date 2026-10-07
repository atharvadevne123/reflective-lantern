"""Feature engineering pipeline for Sports-Oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class FormIndexEncoder(BaseEstimator, TransformerMixin):
    """Encode team form as a weighted rolling win-rate index."""

    def __init__(self, decay: float = 0.9) -> None:
        self.decay = decay

    def fit(self, X: pd.DataFrame, y=None) -> FormIndexEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in ("home_form", "away_form"):
            if col in X.columns:
                X[col] = X[col].fillna(0.5).clip(0.0, 1.0)
        X["form_differential"] = X.get("home_form", 0.5) - X.get("away_form", 0.5)
        X["form_product"] = X.get("home_form", 0.5) * X.get("away_form", 0.5)
        return X


class HeadToHeadEncoder(BaseEstimator, TransformerMixin):
    """Encode head-to-head history ratios."""

    def fit(self, X: pd.DataFrame, y=None) -> HeadToHeadEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        total = (
            X.get("h2h_home_wins", pd.Series(0, index=X.index)).fillna(0)
            + X.get("h2h_draws", pd.Series(0, index=X.index)).fillna(0)
            + X.get("h2h_away_wins", pd.Series(0, index=X.index)).fillna(0)
            + 1e-6
        )
        X["h2h_home_rate"] = X.get("h2h_home_wins", pd.Series(0, index=X.index)).fillna(0) / total
        X["h2h_draw_rate"] = X.get("h2h_draws", pd.Series(0, index=X.index)).fillna(0) / total
        X["h2h_away_rate"] = X.get("h2h_away_wins", pd.Series(0, index=X.index)).fillna(0) / total
        X["h2h_home_dominance"] = X["h2h_home_rate"] - X["h2h_away_rate"]
        return X


class AttackDefenseRatioEncoder(BaseEstimator, TransformerMixin):
    """Compute attack/defense ratio and expected goal proxies."""

    def fit(self, X: pd.DataFrame, y=None) -> AttackDefenseRatioEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        eps = 1e-6
        X["home_attack"] = X.get("home_attack", pd.Series(1.0, index=X.index)).fillna(1.0)
        X["away_attack"] = X.get("away_attack", pd.Series(1.0, index=X.index)).fillna(1.0)
        X["home_defense"] = X.get("home_defense", pd.Series(1.0, index=X.index)).fillna(1.0)
        X["away_defense"] = X.get("away_defense", pd.Series(1.0, index=X.index)).fillna(1.0)
        X["xg_home"] = X["home_attack"] / (X["away_defense"] + eps)
        X["xg_away"] = X["away_attack"] / (X["home_defense"] + eps)
        X["xg_differential"] = X["xg_home"] - X["xg_away"]
        X["attack_ratio"] = X["home_attack"] / (X["away_attack"] + eps)
        X["defense_ratio"] = X["home_defense"] / (X["away_defense"] + eps)
        return X


class RestDayEncoder(BaseEstimator, TransformerMixin):
    """Encode rest days and fatigue penalty."""

    FATIGUE_THRESHOLD = 4

    def fit(self, X: pd.DataFrame, y=None) -> RestDayEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        home_rest = X.get("home_rest_days", pd.Series(7, index=X.index)).fillna(7).clip(1, 14)
        away_rest = X.get("away_rest_days", pd.Series(7, index=X.index)).fillna(7).clip(1, 14)
        X["home_rest_days"] = home_rest
        X["away_rest_days"] = away_rest
        X["rest_differential"] = home_rest - away_rest
        X["home_fatigued"] = (home_rest < self.FATIGUE_THRESHOLD).astype(int)
        X["away_fatigued"] = (away_rest < self.FATIGUE_THRESHOLD).astype(int)
        return X


class DropCategoricalColumns(BaseEstimator, TransformerMixin):
    """Drop non-numeric columns before scaling."""

    def fit(self, X: pd.DataFrame, y=None) -> DropCategoricalColumns:
        self.numeric_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.numeric_cols_].values.astype(float)


def build_feature_pipeline() -> Pipeline:
    """Return the 5-stage sklearn feature engineering pipeline."""
    return Pipeline(
        [
            ("form", FormIndexEncoder()),
            ("h2h", HeadToHeadEncoder()),
            ("attack_defense", AttackDefenseRatioEncoder()),
            ("rest", RestDayEncoder()),
            ("drop_cat", DropCategoricalColumns()),
        ]
    )


LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.classes_ = np.array(["A", "D", "H"])


def make_synthetic_dataset(n: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a signal-bearing synthetic sports dataset for testing."""
    rng = np.random.default_rng(seed)

    home_form = rng.beta(5, 3, n)
    away_form = rng.beta(3, 5, n)
    home_attack = rng.gamma(2, 0.5, n) + 0.5
    away_attack = rng.gamma(2, 0.5, n) + 0.5
    home_defense = rng.gamma(2, 0.5, n) + 0.5
    away_defense = rng.gamma(2, 0.5, n) + 0.5
    h2h_home_wins = rng.integers(0, 10, n)
    h2h_draws = rng.integers(0, 5, n)
    h2h_away_wins = rng.integers(0, 10, n)
    home_rest_days = rng.integers(2, 14, n)
    away_rest_days = rng.integers(2, 14, n)

    xg_home = home_attack / (away_defense + 1e-6)
    xg_away = away_attack / (home_defense + 1e-6)
    h2h_total = h2h_home_wins + h2h_draws + h2h_away_wins + 1e-6
    h2h_home_rate = h2h_home_wins / h2h_total
    h2h_away_rate = h2h_away_wins / h2h_total

    logit = (
        1.5 * (xg_home - xg_away)
        + 2.0 * (home_form - away_form)
        + 1.0 * (h2h_home_rate - h2h_away_rate)
        + 0.3 * (home_rest_days - away_rest_days) / 14.0
        + 0.8  # home advantage constant
        + rng.normal(0, 0.4, n)
    )

    p_home = 1.0 / (1.0 + np.exp(-logit))
    p_draw = 0.28 * (1.0 - np.abs(logit) / 5.0).clip(0.0, 1.0)
    p_away = np.maximum(0.0, 1.0 - p_home - p_draw)
    total = p_home + p_draw + p_away + 1e-9
    p_home /= total
    p_draw /= total
    p_away /= total

    probs = np.stack([p_home, p_draw, p_away], axis=1)
    outcome_idx = np.array([rng.choice(3, p=probs[i]) for i in range(n)])
    labels = np.array(["H", "D", "A"])[outcome_idx]

    X = pd.DataFrame(
        {
            "home_form": home_form,
            "away_form": away_form,
            "home_attack": home_attack,
            "away_attack": away_attack,
            "home_defense": home_defense,
            "away_defense": away_defense,
            "h2h_home_wins": h2h_home_wins,
            "h2h_draws": h2h_draws,
            "h2h_away_wins": h2h_away_wins,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
        }
    )
    y = LABEL_ENCODER.transform(labels)
    return X, y

__all__ = [
    "FormIndexEncoder",
    "HeadToHeadEncoder",
    "AttackDefenseRatioEncoder",
    "RestDayEncoder",
    "DropCategoricalColumns",
    "build_feature_pipeline",
    "make_synthetic_dataset",
    "LABEL_ENCODER",
]
