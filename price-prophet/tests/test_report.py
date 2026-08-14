"""Tests for app/evaluation/report.py."""

from __future__ import annotations


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
