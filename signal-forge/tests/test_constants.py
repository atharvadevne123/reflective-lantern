"""Tests for shared constants."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.constants import (
    ANNUALISATION_FACTOR,
    FEATURE_NAMES,
    MODEL_VERSION,
    REGIME_RISK_WEIGHTS,
    REGIMES,
    RISK_TIERS,
)


def test_regimes_count():
    assert len(REGIMES) == 4


def test_all_regimes_in_weights():
    assert set(REGIMES) == set(REGIME_RISK_WEIGHTS.keys())


def test_regime_weights_range():
    for weight in REGIME_RISK_WEIGHTS.values():
        assert 0.0 <= weight <= 1.0


def test_risk_tiers_cover_zero_to_one():
    for lo, hi in RISK_TIERS.values():
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert lo < hi


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 5


def test_annualisation_factor():
    import math
    assert ANNUALISATION_FACTOR == pytest.approx(math.sqrt(252), rel=1e-5)


@pytest.mark.parametrize("name", ["volatility", "momentum", "volume_ratio", "correlation", "beta"])
def test_all_feature_names_present(name):
    assert name in FEATURE_NAMES


def test_model_version_format():
    parts = MODEL_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
