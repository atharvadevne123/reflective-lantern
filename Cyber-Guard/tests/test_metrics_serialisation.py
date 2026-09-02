"""Tests that training metrics are always valid, parseable JSON.

A CV fold containing none of a rare class yields a NaN AUC for that fold.
``json.dump`` writes bare ``NaN``, which ``json.loads`` accepts but strict
parsers (jq, Go, most JSON schema validators) reject -- so a downstream
dashboard reading metrics.json would break on exactly the small or skewed
training runs where the metric matters most.
"""

from __future__ import annotations

import json

import pytest

from app.model import generate_synthetic_data, train_model


def _train(tmp_path, n_samples, seed=42):
    mp = tmp_path / "m.joblib"
    metp = tmp_path / "metrics.json"
    X, y = generate_synthetic_data(n_samples, seed=seed)
    _, metrics = train_model(X, y, model_path=str(mp), metrics_path=str(metp))
    return metrics, metp


@pytest.mark.parametrize("n_samples", [80, 200, 500])
def test_metrics_json_is_strict_parseable(tmp_path, n_samples):
    """metrics.json must never contain NaN or Infinity at any sample size."""
    _, metp = _train(tmp_path, n_samples)
    raw = metp.read_text()
    assert "NaN" not in raw, "metrics.json contains NaN, which is not valid JSON"
    assert "Infinity" not in raw
    json.loads(raw)  # must parse


def test_auc_is_none_or_float_never_nan(tmp_path):
    metrics, _ = _train(tmp_path, 100)
    auc = metrics["auc_mean"]
    assert auc is None or isinstance(auc, float)
    if isinstance(auc, float):
        assert auc == auc, "auc_mean is NaN"


def test_scored_folds_reported(tmp_path):
    """The metrics record how many folds actually produced an AUC."""
    metrics, _ = _train(tmp_path, 400)
    assert 0 <= metrics["auc_scored_folds"] <= metrics["cv_folds"]


def test_model_learns_real_signal(tmp_path):
    """Class-conditional generation must yield a genuinely learnable task.

    Guards the regression where features and labels were sampled
    independently, capping AUC at chance (~0.5) however good the model was.
    """
    metrics, _ = _train(tmp_path, 600)
    assert metrics["auc_mean"] is not None
    assert metrics["auc_mean"] > 0.75, (
        f"AUC {metrics['auc_mean']:.3f} is near chance — labels look "
        "independent of features"
    )
    assert metrics["accuracy_mean"] > 0.75
