"""Tests for app/utils/serialization.py."""

from __future__ import annotations

import os
import tempfile

import pytest


def test_save_and_load_model():
    from app.utils.serialization import load_model, save_model

    obj = {"key": "value", "number": 42}
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_model(obj, path)
        loaded = load_model(path)
        assert loaded == obj
    finally:
        os.unlink(path)


def test_load_model_missing_file():
    from app.utils.serialization import load_model

    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/model.joblib")


def test_save_model_creates_dirs():
    import tempfile

    from app.utils.serialization import save_model

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "subdir", "model.joblib")
        save_model({"x": 1}, path)
        assert os.path.exists(path)
