"""Tests for model save/load persistence."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestSaveLoad:
    def test_save_after_training(self, trained_models, tmp_path):
        from app.model import save_models
        saved = save_models(tmp_path)
        assert "isolation_forest" in saved
        assert "scaler" in saved
        assert Path(saved["isolation_forest"]).exists()

    def test_load_restores_scoring(self, trained_models, tmp_path):
        from app import model as m
        from app.model import load_models, save_models, score_anomaly
        save_models(tmp_path)
        X = trained_models["X"]
        scores_before = score_anomaly(X, trained_models["if_model"], trained_models["scaler"])

        m._IF_MODEL = None
        m._SCALER = None
        assert load_models(tmp_path) is True

        scores_after = score_anomaly(X)
        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)

    def test_load_from_empty_dir_returns_false(self, tmp_path):
        from app.model import load_models
        assert load_models(tmp_path / "nonexistent") is False

    def test_save_creates_directory(self, trained_models, tmp_path):
        from app.model import save_models
        target = tmp_path / "nested" / "models"
        save_models(target)
        assert target.exists()
