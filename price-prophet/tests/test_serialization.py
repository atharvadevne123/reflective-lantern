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


def test_load_model_file_not_found_raises() -> None:
    import pytest

    from app.utils.serialization import load_model

    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/path/model.joblib")


def test_save_and_load_dict_roundtrip() -> None:
    import os
    import tempfile

    from app.utils.serialization import load_model, save_model

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        obj = {"key": "value", "number": 42}
        save_model(obj, path)
        loaded = load_model(path)
        assert loaded["key"] == "value"
        assert loaded["number"] == 42


def test_save_and_load_list() -> None:
    from app.utils.serialization import load_model, save_model

    obj = [1, 2, 3, 4, 5]
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_model(obj, path)
        loaded = load_model(path)
        assert loaded == obj
    finally:
        os.unlink(path)


def test_save_overwrites_existing_file() -> None:
    from app.utils.serialization import load_model, save_model

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_model({"v": 1}, path)
        save_model({"v": 2}, path)
        loaded = load_model(path)
        assert loaded["v"] == 2
    finally:
        os.unlink(path)


def test_save_empty_dict() -> None:
    from app.utils.serialization import load_model, save_model

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_model({}, path)
        loaded = load_model(path)
        assert loaded == {}
    finally:
        os.unlink(path)


@pytest.mark.parametrize("payload", [{"x": 1}, [1, 2, 3], "just a string", 42])
def test_save_and_load_various_types(payload) -> None:
    from app.utils.serialization import load_model, save_model

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_model(payload, path)
        loaded = load_model(path)
        assert loaded == payload
    finally:
        os.unlink(path)


def test_load_from_nonexistent_path_raises() -> None:
    from app.utils.serialization import load_model

    with pytest.raises(Exception):
        load_model("/nonexistent/path/model.joblib")
