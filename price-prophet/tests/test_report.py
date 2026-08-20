"""Tests for app/evaluation/report.py."""

from __future__ import annotations

import pytest


def test_format_metrics_report_contains_title():
    from app.evaluation.report import format_metrics_report

    report = format_metrics_report({"mae": 1.5, "rmse": 2.1}, title="Test Report")
    assert "Test Report" in report


def test_format_metrics_report_contains_values():
    from app.evaluation.report import format_metrics_report

    report = format_metrics_report({"mae": 1.5})
    assert "mae" in report
    assert "1.5" in report


def test_build_backtest_report_contains_uplift():
    from app.evaluation.report import build_backtest_report

    result = {
        "baseline_revenue": 1000.0,
        "optimised_revenue": 1100.0,
        "revenue_uplift_pct": 10.0,
        "mae": 0.5,
        "n_periods": 5,
    }
    report = build_backtest_report(result)
    assert "uplift" in report.lower() or "10" in report
    assert "5" in report


def test_build_backtest_report_zero_uplift():
    from app.evaluation.report import build_backtest_report

    result = {
        "baseline_revenue": 100.0,
        "optimised_revenue": 100.0,
        "revenue_uplift_pct": 0.0,
        "mae": 0.0,
        "n_periods": 1,
    }
    report = build_backtest_report(result)
    assert isinstance(report, str)


def test_build_backtest_report_contains_uplift_pct() -> None:
    from app.evaluation.report import build_backtest_report

    result = {
        "baseline_revenue": 100.0,
        "optimised_revenue": 120.0,
        "revenue_uplift_pct": 20.0,
        "mae": 5.0,
        "n_periods": 10,
    }
    report = build_backtest_report(result)
    assert "uplift" in report.lower() or "20" in report


def test_build_backtest_report_returns_non_empty() -> None:
    from app.evaluation.report import build_backtest_report

    result = {
        "baseline_revenue": 50.0,
        "optimised_revenue": 60.0,
        "revenue_uplift_pct": 10.0,
        "mae": 2.0,
        "n_periods": 5,
    }
    report = build_backtest_report(result)
    assert len(report) > 0


def test_compare_models_ranks_by_mae() -> None:
    from app.evaluation.report import compare_models

    results = {
        "model_a": {"mae": 2.0},
        "model_b": {"mae": 0.5},
        "model_c": {"mae": 1.0},
    }
    ranks = compare_models(results)
    assert ranks["model_b"] == 1
    assert ranks["model_a"] == 3


def test_compare_models_empty() -> None:
    from app.evaluation.report import compare_models

    assert compare_models({}) == {}


def test_compare_models_by_metric_sorted() -> None:
    from app.evaluation.report import compare_models_by_metric

    results = {
        "a": {"rmse": 3.0},
        "b": {"rmse": 1.0},
        "c": {"rmse": 2.0},
    }
    ranked = compare_models_by_metric(results, metric="rmse")
    names = [name for name, _ in ranked]
    assert names[0] == "b"


def test_metrics_diff_shared_keys() -> None:
    from app.evaluation.report import metrics_diff

    a = {"mae": 2.0, "rmse": 3.0}
    b = {"mae": 1.5, "rmse": 2.5}
    diff = metrics_diff(a, b)
    assert diff["mae"] == pytest.approx(0.5)
    assert diff["rmse"] == pytest.approx(0.5)


def test_metrics_diff_missing_key_excluded() -> None:
    from app.evaluation.report import metrics_diff

    a = {"mae": 1.0, "extra": 5.0}
    b = {"mae": 0.5}
    diff = metrics_diff(a, b)
    assert "extra" not in diff
    assert "mae" in diff


def test_generate_summary_contains_equals_lines() -> None:
    from app.evaluation.report import generate_summary

    summary = generate_summary({"mae": 0.5, "n_samples": 100})
    assert "=" in summary
    assert "mae" in summary.lower() or "MAE" in summary
