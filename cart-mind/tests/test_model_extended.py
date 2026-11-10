"""Extended tests for cart-mind app/model.py model_fingerprint and predict_intent."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from app.model import model_fingerprint, predict_intent


def _make_simple_pipeline() -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 1, 0, 1])
    pipe.fit(X, y)
    return pipe


class TestModelFingerprint:
    def test_returns_dict(self) -> None:
        pipe = _make_simple_pipeline()
        fp = model_fingerprint(pipe)
        assert isinstance(fp, dict)

    def test_has_steps(self) -> None:
        pipe = _make_simple_pipeline()
        fp = model_fingerprint(pipe)
        assert "steps" in fp
        assert "scaler" in fp["steps"]
        assert "clf" in fp["steps"]

    def test_n_steps_correct(self) -> None:
        pipe = _make_simple_pipeline()
        fp = model_fingerprint(pipe)
        assert fp["n_steps"] == 2

    def test_has_param_hash(self) -> None:
        pipe = _make_simple_pipeline()
        fp = model_fingerprint(pipe)
        assert "param_hash" in fp
        assert len(fp["param_hash"]) > 0

    def test_identical_pipelines_same_hash(self) -> None:
        pipe1 = _make_simple_pipeline()
        pipe2 = _make_simple_pipeline()
        assert model_fingerprint(pipe1)["param_hash"] == model_fingerprint(pipe2)["param_hash"]


class TestPredictIntent:
    def test_returns_probabilities_and_labels(self) -> None:
        pipe = _make_simple_pipeline()
        X = pd.DataFrame({"a": [1.5, 2.5], "b": [0.15, 0.25]})
        probs, labels = predict_intent(pipe, X)
        assert len(probs) == 2
        assert len(labels) == 2

    def test_probabilities_in_unit_interval(self) -> None:
        pipe = _make_simple_pipeline()
        X = pd.DataFrame({"a": [1.0, 3.0], "b": [0.1, 0.3]})
        probs, _ = predict_intent(pipe, X)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_labels_binary(self) -> None:
        pipe = _make_simple_pipeline()
        X = pd.DataFrame({"a": [1.0, 3.0], "b": [0.1, 0.3]})
        _, labels = predict_intent(pipe, X)
        assert all(label in (0, 1) for label in labels)
