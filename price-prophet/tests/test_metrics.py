"""Tests for app/evaluation/metrics.py."""
from __future__ import annotations

import math

import pytest


def test_mae_perfect():
    from app.evaluation.metrics import mae
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_basic():
    from app.evaluation.metrics import mae
    result = mae([0.0, 10.0], [5.0, 5.0])
    assert abs(result - 5.0) < 1e-9


def test_mae_empty():
    from app.evaluation.metrics import mae
    with pytest.raises(ValueError):
        mae([], [])


def test_rmse_basic():
    from app.evaluation.metrics import rmse
    result = rmse([0.0, 0.0, 0.0], [3.0, 4.0, 5.0])
    expected = math.sqrt((9 + 16 + 25) / 3)
    assert abs(result - expected) < 1e-9


def test_mape_basic():
    from app.evaluation.metrics import mape
    result = mape([100.0, 200.0], [90.0, 220.0])
    assert result > 0


def test_r_squared_perfect():
    from app.evaluation.metrics import r_squared
    assert r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_revenue_uplift_basic():
    from app.evaluation.metrics import revenue_uplift
    uplift = revenue_uplift(100.0, 110.0)
    assert abs(uplift - 10.0) < 1e-9


def test_revenue_uplift_zero_baseline():
    from app.evaluation.metrics import revenue_uplift
    assert revenue_uplift(0.0, 100.0) == 0.0
