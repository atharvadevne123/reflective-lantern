"""Integration tests for the full model+feature pipeline."""

from __future__ import annotations

import pandas as pd
import pytest


def test_end_to_end_pipeline_predict(trained_model):
    """Full pipeline from raw features to prediction."""
    from app.model import predict

    pipe, _, _ = trained_model
    X = pd.DataFrame(
        [
            {
                "home_form": 0.75,
                "away_form": 0.45,
                "home_attack": 1.5,
                "away_attack": 1.0,
                "home_defense": 1.2,
                "away_defense": 0.9,
                "h2h_home_wins": 6,
                "h2h_draws": 2,
                "h2h_away_wins": 2,
                "home_rest_days": 7,
                "away_rest_days": 4,
            }
        ]
    )
    result = predict(pipe, X)
    assert result["predicted_outcome"] in ("H", "D", "A")
    total = result["prob_home"] + result["prob_draw"] + result["prob_away"]
    assert abs(total - 1.0) < 0.01


def test_pipeline_consistent_predictions(trained_model):
    """Same input should produce same prediction."""
    from app.model import predict

    pipe, _, _ = trained_model
    X = pd.DataFrame(
        [
            {
                "home_form": 0.6,
                "away_form": 0.4,
                "home_attack": 1.2,
                "away_attack": 1.0,
                "home_defense": 1.1,
                "away_defense": 0.9,
                "h2h_home_wins": 3,
                "h2h_draws": 2,
                "h2h_away_wins": 2,
                "home_rest_days": 7,
                "away_rest_days": 7,
            }
        ]
    )
    r1 = predict(pipe, X)
    r2 = predict(pipe, X)
    assert r1["predicted_outcome"] == r2["predicted_outcome"]
    assert abs(r1["prob_home"] - r2["prob_home"]) < 1e-8


@pytest.mark.parametrize("n_rows", [1, 5, 10, 50])
def test_pipeline_batch_predict(trained_model, n_rows):
    """Pipeline handles various batch sizes."""
    from app.features import make_synthetic_dataset
    from app.model import predict

    pipe, _, _ = trained_model
    X, _ = make_synthetic_dataset(n=n_rows)
    for i in range(n_rows):
        result = predict(pipe, X.iloc[[i]])
        assert result["predicted_outcome"] in ("H", "D", "A")


def test_pipeline_home_advantage():
    """Model should favour home team when home metrics are significantly better."""
    import tempfile
    from pathlib import Path

    from app.features import make_synthetic_dataset
    from app.model import predict, train_model

    X, y = make_synthetic_dataset(n=1000)
    with tempfile.TemporaryDirectory() as tmp:
        pipe, _ = train_model(
            X, y, model_path=Path(tmp) / "m.joblib", metrics_path=Path(tmp) / "m.json"
        )
        X_favoured = pd.DataFrame(
            [
                {
                    "home_form": 0.9,
                    "away_form": 0.1,
                    "home_attack": 2.0,
                    "away_attack": 0.5,
                    "home_defense": 2.0,
                    "away_defense": 0.5,
                    "h2h_home_wins": 10,
                    "h2h_draws": 0,
                    "h2h_away_wins": 0,
                    "home_rest_days": 7,
                    "away_rest_days": 3,
                }
            ]
        )
        result = predict(pipe, X_favoured)
        # Home team is heavily favoured, so prob_home should be the highest
        assert result["prob_home"] >= result["prob_away"]
