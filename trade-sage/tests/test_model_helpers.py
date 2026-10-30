"""Tests for model_is_loaded and safe_predict helpers."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from app.model import model_is_loaded, safe_predict


class TestModelIsLoaded:
    def test_returns_false_when_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "missing.pkl")
            with patch("app.model._model_path", return_value=path):
                assert model_is_loaded() is False

    def test_returns_true_when_file_exists(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            with patch("app.model._model_path", return_value=f.name):
                assert model_is_loaded() is True

    def test_returns_bool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "absent.pkl")
            with patch("app.model._model_path", return_value=path):
                result = model_is_loaded()
                assert isinstance(result, bool)


class TestSafePredict:
    def test_returns_predictions_on_success(self) -> None:
        pipe = MagicMock()
        pipe.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
        preds, confs = safe_predict(pipe, np.zeros((2, 5)))
        assert preds == [1, 0]
        assert len(confs) == 2

    def test_returns_none_on_exception(self) -> None:
        pipe = MagicMock()
        pipe.predict_proba.side_effect = RuntimeError("model broken")
        result = safe_predict(pipe, np.zeros((1, 5)))
        assert result is None

    def test_handles_value_error(self) -> None:
        pipe = MagicMock()
        pipe.predict_proba.side_effect = ValueError("shape mismatch")
        assert safe_predict(pipe, np.zeros((1, 5))) is None

    def test_confidences_between_zero_and_one(self) -> None:
        pipe = MagicMock()
        pipe.predict_proba.return_value = np.array([[0.45, 0.55]])
        _, confs = safe_predict(pipe, np.zeros((1, 5)))
        assert all(0.0 <= c <= 1.0 for c in confs)
