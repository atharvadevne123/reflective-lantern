"""Reporting tests for Risk-Sense."""

from __future__ import annotations

import pytest

from app.reporting import (
    export_predictions_csv,
    generate_model_report,
    generate_risk_summary,
)


class TestGenerateRiskSummary:
    def test_empty_returns_zero_count(self) -> None:
        result = generate_risk_summary([])
        assert result["count"] == 0

    def test_single_prediction(self) -> None:
        result = generate_risk_summary([0.5])
        assert result["count"] == 1
        assert result["mean"] == pytest.approx(0.5)

    def test_distribution_sums_to_total(self) -> None:
        predictions = [0.1, 0.3, 0.6, 0.8]
        result = generate_risk_summary(predictions)
        dist = result["risk_distribution"]
        total = dist["low"] + dist["medium"] + dist["high"] + dist["very_high"]
        assert total == 4

    def test_default_rate_calculated(self) -> None:
        predictions = [0.1, 0.9, 0.9, 0.1]  # 2/4 above 0.5
        result = generate_risk_summary(predictions)
        assert result["default_rate"] == pytest.approx(0.5)

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_various_sizes(self, n: int) -> None:
        predictions = [0.3] * n
        result = generate_risk_summary(predictions)
        assert result["count"] == n

    def test_all_high_risk(self) -> None:
        predictions = [0.9] * 10
        result = generate_risk_summary(predictions)
        assert result["risk_distribution"]["very_high"] == 10
        assert result["default_rate"] == pytest.approx(1.0)


class TestGenerateModelReport:
    def test_contains_required_sections(self) -> None:
        metrics = {"n_samples": 1000, "positive_rate": 0.3, "train_auc": 0.85, "cv_auc_mean": 0.83, "cv_auc_std": 0.02}
        report = generate_model_report(metrics)
        assert "Training samples" in report
        assert "Train AUC-ROC" in report
        assert "5-fold CV AUC" in report

    def test_returns_string(self) -> None:
        report = generate_model_report({})
        assert isinstance(report, str)


class TestExportPredictionsCsv:
    def test_empty_returns_header_only(self) -> None:
        csv = export_predictions_csv([])
        lines = csv.strip().split("\n")
        assert len(lines) == 1
        assert "applicant_id" in lines[0]

    def test_single_record(self) -> None:
        records = [{"applicant_id": "test-001", "risk_score": 0.3, "risk_category": "low",
                    "default_probability": 0.3, "created_at": "2026-11-11"}]
        csv = export_predictions_csv(records)
        lines = csv.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "test-001" in lines[1]

    def test_multiple_records(self) -> None:
        records = [
            {"applicant_id": f"app-{i}", "risk_score": 0.3, "risk_category": "low",
             "default_probability": 0.3, "created_at": "2026-11-11"}
            for i in range(5)
        ]
        csv = export_predictions_csv(records)
        lines = [line for line in csv.strip().split("\n") if line]
        assert len(lines) == 6  # header + 5 data rows
