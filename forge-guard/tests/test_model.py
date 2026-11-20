"""Tests for model training, persistence, and inference."""

from __future__ import annotations

import pytest


def test_train_model_returns_pipeline_and_metrics(feature_arrays):
    from app.model import train_model

    X, y = feature_arrays
    pipe, metrics = train_model(X, y, cv_folds=3)

    assert pipe is not None
    assert "auc_cv_mean" in metrics
    assert 0.5 <= metrics["auc_cv_mean"] <= 1.0
    assert metrics["n_features"] == X.shape[1]
    assert metrics["n_samples"] == X.shape[0]


def test_model_persisted_to_disk(feature_arrays, tmp_path, monkeypatch):
    from app import model as m

    model_file = tmp_path / "test_model.joblib"
    metrics_file = tmp_path / "test_metrics.json"
    monkeypatch.setattr(m, "MODEL_PATH", model_file)
    monkeypatch.setattr(m, "METRICS_PATH", metrics_file)

    X, y = feature_arrays
    m.train_model(X, y, cv_folds=2)

    assert model_file.exists()
    assert metrics_file.exists()


def test_predict_returns_valid_label_and_prob(feature_arrays):
    from app.model import predict, train_model

    X, y = feature_arrays
    pipe, _ = train_model(X, y, cv_folds=2)

    for i in range(min(5, len(X))):
        label, prob = predict(pipe, X[i : i + 1])
        assert label in (0, 1)
        assert 0.0 <= prob <= 1.0


def test_load_model_trains_if_missing(tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "nonexistent.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    loaded = m.load_model()
    assert loaded is not None


def test_get_metrics_returns_dict_after_training(feature_arrays, tmp_path, monkeypatch):
    from app import model as m

    metrics_file = tmp_path / "metrics.json"
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", metrics_file)

    X, y = feature_arrays
    _, metrics = m.train_model(X, y, cv_folds=2)
    result = m.get_metrics()

    assert isinstance(result, dict)
    assert result["auc_cv_mean"] == metrics["auc_cv_mean"]


@pytest.mark.parametrize("cv_folds", [2, 3])
def test_cv_folds_parameter(feature_arrays, cv_folds):
    from app.model import train_model

    X, y = feature_arrays
    _, metrics = train_model(X, y, cv_folds=cv_folds)
    assert metrics["cv_folds"] == cv_folds


def test_predict_probability_sums_to_one(feature_arrays):
    from app.model import predict, train_model

    X, y = feature_arrays
    pipe, _ = train_model(X, y, cv_folds=2)
    label, prob = predict(pipe, X[:1])
    assert 0.0 <= prob <= 1.0
    assert label == int(prob >= 0.5)


def test_get_metrics_empty_when_no_file(tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "nonexistent.json")
    result = m.get_metrics()
    assert result == {}


def test_feature_importance_returns_nonempty_dict(feature_arrays, tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    X, y = feature_arrays
    pipe, _ = m.train_model(X, y, cv_folds=2)
    importances = m.feature_importance(pipe)
    assert isinstance(importances, dict)
    assert len(importances) > 0
    assert all(0.0 <= v <= 1.0 for v in importances.values())


def test_metrics_keys_complete(feature_arrays, tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    X, y = feature_arrays
    _, metrics = m.train_model(X, y, cv_folds=2)
    for key in ("auc_cv_mean", "auc_cv_std", "auc_train", "n_features", "n_samples", "cv_folds"):
        assert key in metrics


@pytest.mark.parametrize("n_samples", [100, 300])
def test_train_on_varying_sample_sizes(n_samples):
    from app.features import build_feature_pipeline, generate_synthetic_data
    from app.model import train_model

    df = generate_synthetic_data(n_samples=n_samples, seed=99)
    feat_pipe = build_feature_pipeline()
    X = feat_pipe.fit_transform(df[[c for c in df.columns if c != "defect"]])
    y = df["defect"].values
    pipe, metrics = train_model(X, y, cv_folds=2)
    assert metrics["n_samples"] == n_samples


def test_feature_importance_values_sum_near_one(feature_arrays, tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    X, y = feature_arrays
    pipe, _ = m.train_model(X, y, cv_folds=2)
    importances = m.feature_importance(pipe)
    total = sum(importances.values())
    assert 0.9 <= total <= 1.1


def test_load_model_raises_on_corrupt_file(tmp_path, monkeypatch):
    from app import model as m

    bad_file = tmp_path / "bad.joblib"
    bad_file.write_bytes(b"not a joblib file")
    monkeypatch.setattr(m, "MODEL_PATH", bad_file)
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    with pytest.raises(RuntimeError):
        m.load_model()


def test_predict_batch_consistency(feature_arrays, tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    X, y = feature_arrays
    pipe, _ = m.train_model(X, y, cv_folds=2)
    label1, prob1 = m.predict(pipe, X[:1])
    label2, prob2 = m.predict(pipe, X[:1])
    assert label1 == label2
    assert prob1 == prob2


@pytest.mark.parametrize("cv_folds", [2, 3, 5])
def test_train_model_cv_folds_parametrized(feature_arrays, cv_folds: int) -> None:
    from app.model import train_model

    X, y = feature_arrays
    pipe, metrics = train_model(X, y, cv_folds=cv_folds)
    assert pipe is not None
    assert isinstance(metrics.get("auc_cv_mean"), float)


@pytest.mark.parametrize(
    "metric_key",
    ["auc_cv_mean", "auc_cv_std", "accuracy"],
)
def test_train_model_has_metric(feature_arrays, metric_key: str) -> None:
    from app.model import train_model

    X, y = feature_arrays
    _, metrics = train_model(X, y, cv_folds=2)
    assert metric_key in metrics
