"""Tests for app/data/features.py."""

from __future__ import annotations


def test_price_ratio_basic():
    from app.data.features import price_ratio

    assert abs(price_ratio(100.0, 50.0) - 2.0) < 1e-9


def test_price_ratio_zero_reference():
    from app.data.features import price_ratio

    assert price_ratio(50.0, 0.0) == 1.0


def test_competitive_gap_positive():
    from app.data.features import competitive_gap

    gap = competitive_gap(110.0, 100.0)
    assert abs(gap - 0.1) < 1e-9


def test_competitive_gap_zero_competitor():
    from app.data.features import competitive_gap

    assert competitive_gap(100.0, 0.0) == 0.0


def test_build_feature_matrix(sample_records):
    from app.data.features import build_feature_matrix

    matrix = build_feature_matrix(sample_records)
    assert len(matrix) == len(sample_records)
    assert len(matrix[0]) == 5


def test_build_feature_matrix_first_row(sample_records):
    from app.data.features import build_feature_matrix

    matrix = build_feature_matrix(sample_records)
    rec = sample_records[0]
    assert matrix[0][0] == float(rec["base_price"])
    assert matrix[0][1] == float(rec["demand"])
