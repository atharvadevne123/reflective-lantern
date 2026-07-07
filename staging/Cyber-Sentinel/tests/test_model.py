"""Tests for the ensemble model training and prediction."""
from __future__ import annotations

import numpy as np
import pytest


def test_train_model_returns_metrics() -> None:
    from app.model import train_model

    metrics = train_model()
    assert "f1_score" in metrics
    assert "cv_mean" in metrics
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_predict_returns_label() -> None:
    from app.model import predict, train_model

    train_model()
    features = [0.0] * 25
    result = predict(features)
    assert "label" in result
    assert result["label"] in ("normal", "attack")


def test_predict_confidence_range() -> None:
    from app.model import predict, train_model

    train_model()
    features = [1.0] * 25
    result = predict(features)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_without_training_raises(monkeypatch, tmp_path) -> None:
    import app.model as model_module
    from app.exceptions import ModelNotTrainedError

    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "missing_model.pkl")
    monkeypatch.setattr(model_module, "LABEL_ENCODER_PATH", tmp_path / "missing_le.pkl")

    with pytest.raises(ModelNotTrainedError):
        model_module.predict([0.0] * 25)


def test_train_produces_estimators() -> None:
    from app.model import train_model

    metrics = train_model()
    assert "estimators" in metrics
    assert "rf" in metrics["estimators"]


def test_train_cv_scores_reasonable() -> None:
    from app.model import train_model

    metrics = train_model()
    assert metrics["cv_mean"] > 0.0
    assert metrics["cv_std"] >= 0.0


def test_predict_class_probabilities_sum_to_one() -> None:
    from app.model import predict, train_model

    train_model()
    result = predict([0.5] * 25)
    probs = result["class_probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4


def test_predict_attack_type_is_valid() -> None:
    from app.model import ATTACK_TYPES, predict, train_model

    train_model()
    result = predict([0.0] * 25)
    assert result["attack_type"] in ATTACK_TYPES


@pytest.mark.parametrize("n_features", [25])
def test_predict_with_correct_feature_length(n_features: int) -> None:
    from app.model import predict, train_model

    train_model()
    features = [0.0] * n_features
    result = predict(features)
    assert "label" in result


def test_clear_model_cache_forces_reload() -> None:
    import app.model as model_module
    from app.model import clear_model_cache, predict, train_model

    train_model()
    predict([0.0] * 25)
    clear_model_cache()
    assert "model" not in model_module._model_cache
    # Cache is empty but the pickle persists on disk, so predict reloads it.
    result = predict([0.0] * 25)
    assert 0.0 <= result["confidence"] <= 1.0
    assert "model" in model_module._model_cache


def test_train_with_custom_data() -> None:
    from app.model import train_model

    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 25))
    y = rng.integers(0, 5, size=200)
    metrics = train_model(X, y)
    assert "f1_score" in metrics


def test_feature_importance_returns_dict() -> None:
    from app.model import get_feature_importance, train_model

    train_model()
    importance = get_feature_importance()
    assert isinstance(importance, dict)
