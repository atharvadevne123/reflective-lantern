"""Model training and prediction tests for Cyber-Guard."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from app.model import (
    THREAT_CLASSES,
    generate_synthetic_data,
    load_model,
    predict,
    train_model,
)


def test_generate_synthetic_data_shape():
    X, y = generate_synthetic_data(200)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 200
    assert len(y) == 200


def test_generate_synthetic_data_classes():
    _, y = generate_synthetic_data(300)
    unique = set(y.unique())
    assert unique.issubset(set(THREAT_CLASSES))


def test_train_model_returns_metrics():
    X, y = generate_synthetic_data(150)
    with tempfile.TemporaryDirectory() as tmpdir:
        mp = os.path.join(tmpdir, "model.joblib")
        metp = os.path.join(tmpdir, "metrics.json")
        _, metrics = train_model(X, y, model_path=mp, metrics_path=metp)
    assert "accuracy_mean" in metrics
    assert 0.0 <= metrics["accuracy_mean"] <= 1.0
    assert metrics["n_features"] == 15


def test_train_and_load_predict():
    X, y = generate_synthetic_data(150)
    with tempfile.TemporaryDirectory() as tmpdir:
        mp = os.path.join(tmpdir, "model.joblib")
        metp = os.path.join(tmpdir, "metrics.json")
        _, _ = train_model(X, y, model_path=mp, metrics_path=metp)
        pipe, le = load_model(mp)

    sample = pd.DataFrame(
        [
            {
                "src_bytes": 100.0,
                "dst_bytes": 50.0,
                "duration": 1.0,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF",
            }
        ]
    )
    result = predict(sample, pipe, le)
    assert result["prediction"] in THREAT_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["class_probabilities"].keys()) == set(THREAT_CLASSES)


@pytest.mark.parametrize("n_samples", [100, 300])
def test_train_model_different_sizes(n_samples: int):
    X, y = generate_synthetic_data(n_samples)
    with tempfile.TemporaryDirectory() as tmpdir:
        mp = os.path.join(tmpdir, "model.joblib")
        metp = os.path.join(tmpdir, "metrics.json")
        _, metrics = train_model(X, y, model_path=mp, metrics_path=metp)
    assert metrics["n_samples"] == n_samples


def test_probabilities_sum_to_one():
    X, y = generate_synthetic_data(150)
    with tempfile.TemporaryDirectory() as tmpdir:
        mp = os.path.join(tmpdir, "model.joblib")
        metp = os.path.join(tmpdir, "metrics.json")
        _, _ = train_model(X, y, model_path=mp, metrics_path=metp)
        pipe, le = load_model(mp)

    sample = pd.DataFrame(
        [
            {
                "src_bytes": 500.0,
                "dst_bytes": 100.0,
                "duration": 0.5,
                "protocol_type": "udp",
                "service": "dns",
                "flag": "SF",
            }
        ]
    )
    result = predict(sample, pipe, le)
    total = sum(result["class_probabilities"].values())
    assert abs(total - 1.0) < 1e-6
