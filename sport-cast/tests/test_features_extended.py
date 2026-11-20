"""Extended feature engineering tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    FatigueTransformer,
    FormEncoder,
    HeadToHeadTransformer,
    make_feature_pipeline,
    make_synthetic_dataset,
)


@pytest.mark.parametrize("seed", [0, 1, 42, 99])
def test_synthetic_dataset_reproducible(seed):
    X1, y1 = make_synthetic_dataset(n=100, seed=seed)
    X2, y2 = make_synthetic_dataset(n=100, seed=seed)
    assert (y1 == y2).all()


def test_form_encoder_all_wins():
    df = pd.DataFrame([{"home_wins_last5": 5, "home_draws_last5": 0, "home_losses_last5": 0,
                         "away_wins_last5": 5, "away_draws_last5": 0, "away_losses_last5": 0}])
    out = FormEncoder().fit_transform(df)
    assert float(out["home_win_rate"].iloc[0]) == pytest.approx(1.0)


def test_fatigue_max_rest_near_zero():
    df = pd.DataFrame([{"home_days_rest": 100, "away_days_rest": 100}])
    out = FatigueTransformer().fit_transform(df)
    assert float(out["home_fatigue"].iloc[0]) < 0.01


def test_h2h_transformer_no_history():
    df = pd.DataFrame([{"h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_draws": 0}])
    out = HeadToHeadTransformer().fit_transform(df)
    assert not out["h2h_home_win_rate"].isna().any()
    assert not out["h2h_away_win_rate"].isna().any()


def test_pipeline_no_nan_output():
    X, _ = make_synthetic_dataset(n=100)
    pipe = make_feature_pipeline()
    out = pipe.fit_transform(X)
    assert not np.isnan(out).any()


def test_pipeline_no_inf_output():
    X, _ = make_synthetic_dataset(n=100)
    pipe = make_feature_pipeline()
    out = pipe.fit_transform(X)
    assert not np.isinf(out).any()
