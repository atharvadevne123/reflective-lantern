"""Model training and prediction tests."""
from __future__ import annotations

import pytest

from app.features import make_feature_pipeline, make_synthetic_dataset
from app.model import (
    OUTCOME_LABELS,
    build_ensemble,
    predict_match,
    score_player_performance,
    train_model,
)


def test_train_model_returns_metrics(synthetic_dataset):
    X, y = synthetic_dataset
    _, _, metrics = train_model(X, y, save=False)
    assert "auc_mean" in metrics
    assert 0.0 < metrics["auc_mean"] <= 1.0
    assert metrics["n_samples"] == len(y)


def test_train_model_auc_above_chance(synthetic_dataset):
    X, y = synthetic_dataset
    _, _, metrics = train_model(X, y, save=False)
    assert metrics["auc_mean"] > 0.55, f"AUC too low: {metrics['auc_mean']}"


def test_predict_match_probabilities_sum_to_one(sample_dataframe):
    X, y = make_synthetic_dataset(n=300, seed=1)
    pipeline = make_feature_pipeline()
    X_t = pipeline.fit_transform(X)
    model = build_ensemble()
    model.fit(X_t, y)
    result = predict_match(sample_dataframe, model, pipeline)
    total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    assert abs(total - 1.0) < 0.01


def test_predict_match_valid_outcome(sample_dataframe):
    X, y = make_synthetic_dataset(n=300, seed=2)
    pipeline = make_feature_pipeline()
    X_t = pipeline.fit_transform(X)
    model = build_ensemble()
    model.fit(X_t, y)
    result = predict_match(sample_dataframe, model, pipeline)
    assert result["predicted_outcome"] in OUTCOME_LABELS.values()


@pytest.mark.parametrize("goals,assists,win_rate,minutes,injury,expected_min", [
    (1.0, 0.5, 0.8, 0.9, 0, 40),
    (0.0, 0.0, 0.0, 0.0, 0, 0),
    (0.5, 0.3, 0.6, 0.8, 0, 20),
])
def test_score_player_performance(goals, assists, win_rate, minutes, injury, expected_min):
    result = score_player_performance(goals, assists, win_rate, minutes, injury)
    assert result["performance_score"] >= expected_min
    assert 0.0 <= result["performance_score"] <= 100.0
    assert 0.0 <= result["fatigue_index"] <= 1.0
    assert 0.0 <= result["form_rating"] <= 1.0


def test_score_player_injury_reduces_score():
    healthy = score_player_performance(0.5, 0.3, 0.6, 0.8, 0)
    injured = score_player_performance(0.5, 0.3, 0.6, 0.8, 10)
    assert healthy["performance_score"] > injured["performance_score"]


def test_build_ensemble_has_three_estimators():
    ensemble = build_ensemble()
    assert len(ensemble.estimators) == 3
