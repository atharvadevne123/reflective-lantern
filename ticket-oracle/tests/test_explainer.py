"""Explainer module tests for Ticket-Oracle."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_rf_pipeline(n_features: int = 10) -> tuple[Pipeline, list[str]]:
    rng = np.random.default_rng(0)
    X = rng.random((100, n_features)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)

    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X, y)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return pipe, feature_names


def test_top_features_returns_top_n():
    from app.explainer import top_features

    names = ["a", "b", "c", "d", "e"]
    importances = np.array([0.1, 0.5, 0.2, 0.8, 0.05])
    result = top_features(names, importances, top_n=3)
    assert len(result) == 3
    assert result[0]["feature"] == "d"


def test_top_features_sorted_descending():
    from app.explainer import top_features

    names = [f"f{i}" for i in range(5)]
    importances = np.array([0.3, 0.1, 0.9, 0.5, 0.2])
    result = top_features(names, importances, top_n=5)
    vals = [r["importance"] for r in result]
    assert vals == sorted(vals, reverse=True)


def test_top_features_mismatch_returns_empty():
    from app.explainer import top_features

    result = top_features(["a", "b"], np.array([0.1, 0.2, 0.3]), top_n=2)
    assert result == []


def test_explain_prediction_rf():
    from app.explainer import explain_prediction

    pipe, names = _make_rf_pipeline()
    result = explain_prediction(pipe, names, top_n=3)
    assert len(result) == 3
    for item in result:
        assert "feature" in item
        assert "importance" in item
        assert item["importance"] >= 0


def test_explain_prediction_no_importance_attr():
    from app.explainer import explain_prediction

    names = ["a", "b"]
    # Pipeline with no importances or coef_
    class DummyClf:
        pass

    pipe = Pipeline([("clf", DummyClf())])
    result = explain_prediction(pipe, names)
    assert result == []


def test_extract_feature_names_missing():
    from app.explainer import extract_feature_names

    class BadPipe:
        named_steps = {}

    result = extract_feature_names(BadPipe())
    assert result == []
