"""Feature engineering pipeline tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    DropCategoricalTransformer,
    FatigueTransformer,
    FormEncoder,
    HeadToHeadTransformer,
    LagRollingTransformer,
    RatioFeatureTransformer,
    make_feature_pipeline,
    make_synthetic_dataset,
)


@pytest.fixture()
def base_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "home_wins_last5": 3, "home_draws_last5": 1, "home_losses_last5": 1,
        "away_wins_last5": 2, "away_draws_last5": 1, "away_losses_last5": 2,
        "home_goals_avg": 1.8, "home_goals_conceded_avg": 1.1,
        "away_goals_avg": 1.4, "away_goals_conceded_avg": 1.5,
        "h2h_home_wins": 3, "h2h_away_wins": 2, "h2h_draws": 1,
        "home_ranking": 5.0, "away_ranking": 12.0,
        "home_elo": 1650.0, "away_elo": 1520.0,
        "home_days_rest": 7, "away_days_rest": 4,
        "home_is_home_ground": 1.0,
    }])


def test_form_encoder_produces_win_rate(base_df):
    enc = FormEncoder()
    out = enc.fit_transform(base_df)
    assert "home_win_rate" in out.columns
    assert "away_win_rate" in out.columns
    assert 0.0 <= float(out["home_win_rate"].iloc[0]) <= 1.0


def test_form_encoder_zero_games():
    df = pd.DataFrame([{
        "home_wins_last5": 0, "home_draws_last5": 0, "home_losses_last5": 0,
        "away_wins_last5": 0, "away_draws_last5": 0, "away_losses_last5": 0,
    }])
    enc = FormEncoder()
    out = enc.fit_transform(df)
    assert not out["home_win_rate"].isna().any()


def test_lag_rolling_produces_diff(base_df):
    t = LagRollingTransformer()
    out = t.fit_transform(base_df)
    assert "home_goal_diff_avg" in out.columns
    assert "home_attack_strength" in out.columns


def test_h2h_rates_sum_to_one(base_df):
    t = HeadToHeadTransformer()
    out = t.fit_transform(base_df)
    total = out["h2h_home_win_rate"] + out["h2h_away_win_rate"] + out["h2h_draw_rate"]
    assert abs(float(total.iloc[0]) - 1.0) < 1e-6


def test_ratio_transformer_elo_diff(base_df):
    t = RatioFeatureTransformer()
    out = t.fit_transform(base_df)
    assert "elo_diff" in out.columns
    assert float(out["elo_diff"].iloc[0]) == pytest.approx(1650.0 - 1520.0)


def test_fatigue_transformer_higher_rest_lower_fatigue(base_df):
    t = FatigueTransformer()
    out = t.fit_transform(base_df)
    assert 0.0 <= float(out["home_fatigue"].iloc[0]) <= 1.0
    df_rested = base_df.copy()
    df_rested["home_days_rest"] = 14
    out_rested = t.transform(df_rested)
    assert float(out_rested["home_fatigue"].iloc[0]) < float(out["home_fatigue"].iloc[0])


def test_drop_categorical_returns_ndarray(base_df):
    t = DropCategoricalTransformer()
    out = t.fit_transform(base_df)
    assert isinstance(out, np.ndarray)


def test_full_pipeline_shape(base_df):
    pipe = make_feature_pipeline()
    out = pipe.fit_transform(base_df)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[0] == 1


def test_synthetic_dataset_shape():
    X, y = make_synthetic_dataset(n=100, seed=7)
    assert len(X) == 100
    assert len(y) == 100
    assert set(y).issubset({0, 1, 2})


@pytest.mark.parametrize("n_samples", [50, 200, 500])
def test_pipeline_various_sizes(n_samples):
    X, _ = make_synthetic_dataset(n=n_samples, seed=n_samples)
    pipe = make_feature_pipeline()
    out = pipe.fit_transform(X)
    assert out.shape[0] == n_samples
