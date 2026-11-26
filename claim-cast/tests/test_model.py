"""Model training and prediction tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model import (
    SEVERITY_LABELS,
    _compute_triage,
    _seed_model,
    build_pipeline,
    predict,
)


@pytest.fixture(scope="module")
def trained_model():
    _seed_model()
    return True


def _make_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "claim_age_days": rng.integers(0, 400, n).astype(float),
            "policy_limit": rng.choice([50_000, 100_000, 250_000], n).astype(float),
            "claimed_amount": rng.exponential(20_000, n),
            "num_prior_claims": rng.integers(0, 6, n).astype(float),
            "policy_tenure_years": rng.uniform(1, 15, n),
            "geo_risk_tier": rng.integers(1, 6, n).astype(float),
            "days_since_last_claim": rng.integers(10, 800, n).astype(float),
            "policy_type": rng.choice(["home", "auto", "life", "commercial"], n),
        }
    )


def test_build_pipeline_has_three_steps():
    pipe = build_pipeline()
    assert len(pipe.steps) == 3


def test_seed_model_creates_artifacts(trained_model):
    from pathlib import Path

    assert Path("claim_cast_model.joblib").exists()
    assert Path("claim_cast_anomaly.joblib").exists()


def test_predict_single_row(trained_model):
    X = _make_df(1)
    result = predict(X)
    assert "severity_label" in result
    assert result["severity_label"] in SEVERITY_LABELS.values()
    assert 0.0 <= result["severity_score"] <= 1.0
    assert result["anomaly_flag"] in {0, 1}
    assert result["triage_priority"] in {1, 2, 3, 4}


def test_predict_probabilities_sum_to_one(trained_model):
    X = _make_df(1)
    result = predict(X)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-4


@pytest.mark.parametrize(
    "severity_class,anomaly_flag,expected_priority",
    [
        (3, 0, 1),
        (2, 1, 1),
        (2, 0, 2),
        (1, 0, 3),
        (0, 0, 4),
    ],
)
def test_compute_triage(severity_class, anomaly_flag, expected_priority):
    assert _compute_triage(severity_class, anomaly_flag) == expected_priority


def test_train_model_metrics(trained_model):
    import json
    from pathlib import Path

    metrics = json.loads(Path("claim_cast_metrics.json").read_text())
    assert "f1_macro_mean" in metrics
    assert 0.0 <= metrics["f1_macro_mean"] <= 1.0
    assert metrics["n_features"] > 0


def test_cross_validate_model(trained_model):
    import numpy as np
    import pandas as pd

    from app.model import cross_validate_model

    rng = np.random.default_rng(55)
    n = 200
    X = pd.DataFrame(
        {
            "claim_age_days": rng.integers(0, 400, n).astype(float),
            "policy_limit": rng.choice([50_000, 100_000, 250_000], n).astype(float),
            "claimed_amount": rng.exponential(20_000, n),
            "num_prior_claims": rng.integers(0, 6, n).astype(float),
            "policy_tenure_years": rng.uniform(1, 15, n),
            "geo_risk_tier": rng.integers(1, 6, n).astype(float),
            "days_since_last_claim": rng.integers(10, 800, n).astype(float),
            "policy_type": rng.choice(["home", "auto", "life", "commercial"], n),
        }
    )
    ratio = (X["claimed_amount"] / X["policy_limit"]).clip(0, 1.0)
    y = pd.cut(ratio, bins=[-0.001, 0.1, 0.3, 0.6, 1.001], labels=[0, 1, 2, 3]).astype(int)
    cv_results = cross_validate_model(X, y, cv=3)
    assert cv_results["cv_folds"] == 3
    assert len(cv_results["test_f1_macro"]) == 3
    assert 0.0 <= cv_results["mean_f1_macro"] <= 1.0
