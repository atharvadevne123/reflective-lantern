"""Feature engineering pipeline tests for Sports-Oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _base_row(**kwargs) -> dict:
    defaults = {
        "home_form": 0.6,
        "away_form": 0.4,
        "home_attack": 1.2,
        "away_attack": 1.0,
        "home_defense": 1.1,
        "away_defense": 0.9,
        "h2h_home_wins": 4,
        "h2h_draws": 2,
        "h2h_away_wins": 2,
        "home_rest_days": 7,
        "away_rest_days": 7,
    }
    defaults.update(kwargs)
    return defaults


def test_form_encoder_adds_differential():
    from app.features import FormIndexEncoder

    enc = FormIndexEncoder()
    df = pd.DataFrame([_base_row()])
    out = enc.fit_transform(df)
    assert "form_differential" in out.columns
    assert abs(out["form_differential"].iloc[0] - 0.2) < 1e-6


def test_h2h_encoder_ratios_sum_to_one():
    from app.features import HeadToHeadEncoder

    enc = HeadToHeadEncoder()
    df = pd.DataFrame([_base_row(h2h_home_wins=5, h2h_draws=3, h2h_away_wins=2)])
    out = enc.fit_transform(df)
    total = out["h2h_home_rate"].iloc[0] + out["h2h_draw_rate"].iloc[0] + out["h2h_away_rate"].iloc[0]
    assert abs(total - 1.0) < 1e-4


def test_attack_defense_encoder_adds_xg():
    from app.features import AttackDefenseRatioEncoder

    enc = AttackDefenseRatioEncoder()
    df = pd.DataFrame([_base_row()])
    out = enc.fit_transform(df)
    assert "xg_home" in out.columns
    assert "xg_differential" in out.columns
    assert out["xg_home"].iloc[0] > 0


def test_rest_encoder_flags_fatigue():
    from app.features import RestDayEncoder

    enc = RestDayEncoder()
    df = pd.DataFrame([_base_row(home_rest_days=2, away_rest_days=7)])
    out = enc.fit_transform(df)
    assert out["home_fatigued"].iloc[0] == 1
    assert out["away_fatigued"].iloc[0] == 0


def test_drop_categorical_returns_array():
    from app.features import DropCategoricalColumns

    df = pd.DataFrame([_base_row()])
    enc = DropCategoricalColumns()
    arr = enc.fit_transform(df)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == float


def test_full_pipeline_output_shape():
    from app.features import build_feature_pipeline, make_synthetic_dataset

    X, _ = make_synthetic_dataset(n=50)
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(X)
    assert out.shape[0] == 50
    assert out.shape[1] > 11  # enriched features


def test_full_pipeline_no_nan():
    from app.features import build_feature_pipeline, make_synthetic_dataset

    X, _ = make_synthetic_dataset(n=100)
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(X)
    assert not np.any(np.isnan(out))


@pytest.mark.parametrize("home_form,away_form", [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)])
def test_form_encoder_edge_cases(home_form, away_form):
    from app.features import FormIndexEncoder

    enc = FormIndexEncoder()
    df = pd.DataFrame([_base_row(home_form=home_form, away_form=away_form)])
    out = enc.fit_transform(df)
    assert abs(out["form_differential"].iloc[0] - (home_form - away_form)) < 1e-6


def test_make_synthetic_dataset_shape():
    from app.features import make_synthetic_dataset

    X, y = make_synthetic_dataset(n=200)
    assert len(X) == 200
    assert len(y) == 200
    assert set(np.unique(y)) <= {0, 1, 2}


def test_h2h_encoder_zero_history():
    from app.features import HeadToHeadEncoder

    enc = HeadToHeadEncoder()
    df = pd.DataFrame([_base_row(h2h_home_wins=0, h2h_draws=0, h2h_away_wins=0)])
    out = enc.fit_transform(df)
    # Should not divide by zero
    assert not np.isnan(out["h2h_home_rate"].iloc[0])
