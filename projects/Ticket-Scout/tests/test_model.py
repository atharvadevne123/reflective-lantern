"""Model training and prediction tests."""

from __future__ import annotations

import pytest

from app.model import (
    CATEGORY_LABELS,
    generate_synthetic_data,
    predict,
    train_models,
)


def test_generate_synthetic_data_shape():
    df = generate_synthetic_data(n_samples=100)
    assert len(df) == 100
    assert "subject" in df.columns
    assert "body" in df.columns
    assert "category" in df.columns
    assert "sla_breach" in df.columns
    assert "resolution_hours" in df.columns


def test_generate_synthetic_data_categories():
    df = generate_synthetic_data(n_samples=500)
    cats = set(df["category"].unique())
    assert cats == {"access", "email", "hardware", "network", "software"}


def test_generate_synthetic_data_sla_breach_is_binary():
    df = generate_synthetic_data(n_samples=200)
    assert set(df["sla_breach"].unique()).issubset({0, 1})


def test_train_models_returns_metrics():
    df = generate_synthetic_data(n_samples=300)
    metrics = train_models(df)
    assert "breach_auc_mean" in metrics
    assert "category_accuracy_mean" in metrics
    assert 0.0 <= metrics["breach_auc_mean"] <= 1.0


def test_train_models_auc_above_baseline():
    df = generate_synthetic_data(n_samples=500)
    metrics = train_models(df)
    assert metrics["breach_auc_mean"] > 0.5, "Model should beat random baseline"


def test_predict_returns_correct_structure(trained_models, sample_ticket_df):
    cat_pipe, breach_pipe, res_pipe = trained_models
    results = predict(sample_ticket_df, cat_pipe, breach_pipe, res_pipe)
    assert len(results) == 1
    r = results[0]
    assert "predicted_category" in r
    assert "sla_breach_prob" in r
    assert "resolution_hours_pred" in r
    assert "confidence" in r


@pytest.mark.parametrize("n_tickets", [1, 5, 10])
def test_predict_batch_sizes(trained_models, n_tickets):
    cat_pipe, breach_pipe, res_pipe = trained_models
    df = generate_synthetic_data(n_samples=n_tickets)
    df = df.drop(columns=["category", "sla_breach", "resolution_hours"])
    results = predict(df, cat_pipe, breach_pipe, res_pipe)
    assert len(results) == n_tickets


def test_predict_category_is_valid(trained_models, sample_ticket_df):
    cat_pipe, breach_pipe, res_pipe = trained_models
    results = predict(sample_ticket_df, cat_pipe, breach_pipe, res_pipe)
    assert results[0]["predicted_category"] in CATEGORY_LABELS


def test_predict_probabilities_in_range(trained_models, sample_batch_df):
    cat_pipe, breach_pipe, res_pipe = trained_models
    results = predict(sample_batch_df, cat_pipe, breach_pipe, res_pipe)
    for r in results:
        assert 0.0 <= r["sla_breach_prob"] <= 1.0
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["resolution_hours_pred"] >= 0.0
