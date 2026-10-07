"""Model training and prediction tests for Sports-Oracle."""

from __future__ import annotations

import pytest


def test_train_model_returns_pipeline_and_metrics(tmp_path):
    from app.features import make_synthetic_dataset
    from app.model import train_model

    X, y = make_synthetic_dataset(n=300)
    pipe, metrics = train_model(
        X, y, model_path=tmp_path / "m.joblib", metrics_path=tmp_path / "m.json"
    )
    assert pipe is not None
    assert "auc_mean" in metrics
    assert metrics["auc_mean"] > 0.4
    assert "n_samples" in metrics
    assert metrics["n_samples"] == 300


def test_train_model_persists_files(tmp_path):
    from app.features import make_synthetic_dataset
    from app.model import train_model

    X, y = make_synthetic_dataset(n=200)
    train_model(X, y, model_path=tmp_path / "m.joblib", metrics_path=tmp_path / "m.json")
    assert (tmp_path / "m.joblib").exists()
    assert (tmp_path / "m.json").exists()


def test_load_model_roundtrip(tmp_path):
    from app.features import make_synthetic_dataset
    from app.model import load_model, train_model

    X, y = make_synthetic_dataset(n=200)
    model_path = tmp_path / "m.joblib"
    train_model(X, y, model_path=model_path, metrics_path=tmp_path / "m.json")
    loaded = load_model(model_path)
    assert loaded is not None


def test_load_model_missing_raises(tmp_path):
    from app.model import load_model

    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.joblib")


def test_predict_output_structure(trained_model):

    from app.features import make_synthetic_dataset
    from app.model import predict

    pipe, _, _ = trained_model
    X, _ = make_synthetic_dataset(n=5)
    result = predict(pipe, X.head(1))
    assert result["predicted_outcome"] in ("H", "D", "A")
    assert 0.0 <= result["prob_home"] <= 1.0
    assert 0.0 <= result["prob_draw"] <= 1.0
    assert 0.0 <= result["prob_away"] <= 1.0
    prob_sum = result["prob_home"] + result["prob_draw"] + result["prob_away"]
    assert abs(prob_sum - 1.0) < 0.01


@pytest.mark.parametrize("n_samples", [200, 500, 1000])
def test_train_model_various_sizes(tmp_path, n_samples):
    from app.features import make_synthetic_dataset
    from app.model import train_model

    X, y = make_synthetic_dataset(n=n_samples)
    _, metrics = train_model(
        X,
        y,
        model_path=tmp_path / f"m_{n_samples}.joblib",
        metrics_path=tmp_path / f"m_{n_samples}.json",
    )
    assert metrics["n_samples"] == n_samples
    assert metrics["auc_mean"] > 0.4


def test_read_metrics(tmp_path):
    from app.features import make_synthetic_dataset
    from app.model import read_metrics, train_model

    X, y = make_synthetic_dataset(n=200)
    metrics_path = tmp_path / "metrics.json"
    _, orig = train_model(X, y, model_path=tmp_path / "m.joblib", metrics_path=metrics_path)
    loaded = read_metrics(metrics_path)
    assert loaded["auc_mean"] == orig["auc_mean"]


def test_read_metrics_missing(tmp_path):
    from app.model import read_metrics

    result = read_metrics(tmp_path / "nonexistent.json")
    assert result == {}
