"""
Model serialisation utilities for Price-Prophet.

Uses :mod:`joblib` for efficient persistence of NumPy-heavy model
objects (sklearn, XGBoost, etc.).  Also provides lightweight introspection
helpers used by the model registry and API.
"""

from __future__ import annotations

import os
from typing import Any


def save_model(model: Any, path: str) -> None:
    """Persist *model* to *path* using :func:`joblib.dump`.

    Parent directories are created automatically.

    Parameters
    ----------
    model:
        Any picklable Python object (typically a fitted pricing model).
    path:
        Destination file path (e.g. ``"models/linear_v1.joblib"``).
    """
    import joblib  # type: ignore

    ensure_dir(path)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Load a model previously persisted with :func:`save_model`.

    Parameters
    ----------
    path:
        Path to the serialised model file.

    Returns
    -------
    Any
        The deserialised model object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    """
    import joblib  # type: ignore

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def model_to_dict(model: Any) -> dict:
    """Return a lightweight JSON-serialisable summary of *model*.

    Parameters
    ----------
    model:
        Any model object, fitted or not.

    Returns
    -------
    dict
        ``{"class": str, "fitted": bool}`` where ``class`` is the
        model's type name and ``fitted`` reflects the ``is_fitted()``
        method if available.
    """
    fitted: bool = False
    if hasattr(model, "is_fitted") and callable(model.is_fitted):
        fitted = bool(model.is_fitted())

    return {
        "class": type(model).__name__,
        "fitted": fitted,
    }


def ensure_dir(path: str) -> None:
    """Create all parent directories of *path* if they do not exist.

    Parameters
    ----------
    path:
        A file path whose parent directories should be created.  The
        file itself is not created.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
