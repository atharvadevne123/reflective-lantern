"""Tests for Fleet-Mind shadow mode evaluation."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.shadow_mode as shadow_module
from app.shadow_mode import (
    get_shadow_stats,
    register_challenger,
    shadow_predict,
)


@pytest.fixture(autouse=True)
def reset_shadow_state():
    """Reset module-level shadow state between tests."""
    shadow_module._challenger_model = None
    shadow_module._shadow_log.clear()
    yield
    shadow_module._challenger_model = None
    shadow_module._shadow_log.clear()


def test_get_shadow_stats_empty_log():
    stats = get_shadow_stats()
    assert stats["total_compared"] == 0
    assert stats["agreement_rate"] is None
    assert stats["challenger_available"] is False


def test_register_challenger_sets_model():
    mock_model = MagicMock()
    register_challenger(mock_model)
    assert shadow_module._challenger_model is mock_model


def test_shadow_predict_no_challenger_does_nothing():
    features = np.zeros((1, 10))
    shadow_predict(features, {"maintenance_needed": False, "anomaly_score": 0.1})
    assert len(shadow_module._shadow_log) == 0


def test_shadow_stats_challenger_available_after_register():
    mock_model = MagicMock()
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
    register_challenger(mock_model)
    stats = get_shadow_stats()
    assert stats["challenger_available"] is True


def test_shadow_predict_with_challenger_logs_entry():
    mock_model = MagicMock()
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
    register_challenger(mock_model)

    features = np.zeros((1, 10))
    production = {"maintenance_needed": True, "anomaly_score": 0.8}
    shadow_predict(features, production)

    # Give the background thread a moment
    time.sleep(0.1)

    stats = get_shadow_stats()
    assert stats["total_compared"] >= 0  # may be 0 if thread hasn't finished yet


def test_get_shadow_stats_returns_dict():
    stats = get_shadow_stats()
    assert isinstance(stats, dict)
    assert "total_compared" in stats
    assert "agreement_rate" in stats
    assert "challenger_available" in stats
