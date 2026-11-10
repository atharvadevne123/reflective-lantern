"""Extended tests for ops-vision app/crud.py database operations."""

from __future__ import annotations

import pytest

from app.crud import (
    avg_confidence,
    bulk_create_predictions,
    count_predictions,
    count_predictions_by_service,
    create_prediction,
    delete_old_predictions,
    get_prediction_by_id,
)


class TestCountPredictions:
    def test_zero_on_empty_db(self, db_session) -> None:
        assert count_predictions(db_session) == 0

    def test_increments_after_insert(self, db_session) -> None:
        create_prediction(db_session, {
            "service_name": "web", "cpu_util": 0.5, "mem_util": 0.4,
            "latency_ms": 10.0, "error_rate": 0.0, "throughput": 100.0,
            "predicted_label": "normal", "confidence": 0.9,
        })
        assert count_predictions(db_session) == 1

    def test_counts_multiple(self, db_session) -> None:
        for i in range(3):
            create_prediction(db_session, {
                "service_name": f"svc{i}", "cpu_util": 0.1 * i,
                "mem_util": 0.2, "latency_ms": 10.0, "error_rate": 0.0,
                "throughput": 50.0, "predicted_label": "normal", "confidence": 0.8,
            })
        assert count_predictions(db_session) == 3


class TestAvgConfidence:
    def test_zero_on_empty_db(self, db_session) -> None:
        result = avg_confidence(db_session)
        assert result == pytest.approx(0.0)

    def test_single_entry(self, db_session) -> None:
        create_prediction(db_session, {
            "service_name": "api", "cpu_util": 0.5, "mem_util": 0.5,
            "latency_ms": 5.0, "error_rate": 0.0, "throughput": 100.0,
            "predicted_label": "normal", "confidence": 0.95,
        })
        assert avg_confidence(db_session) == pytest.approx(0.95, abs=1e-3)

    def test_average_of_two(self, db_session) -> None:
        for conf in [0.6, 0.8]:
            create_prediction(db_session, {
                "service_name": "svc", "cpu_util": 0.3, "mem_util": 0.3,
                "latency_ms": 5.0, "error_rate": 0.0, "throughput": 100.0,
                "predicted_label": "normal", "confidence": conf,
            })
        result = avg_confidence(db_session)
        assert result == pytest.approx(0.7, abs=1e-3)


class TestBulkCreatePredictions:
    def test_inserts_all_items(self, db_session) -> None:
        items = [
            {"service_name": f"s{i}", "cpu_util": 0.1, "mem_util": 0.1,
             "latency_ms": 5.0, "error_rate": 0.0, "throughput": 10.0,
             "predicted_label": "normal", "confidence": 0.9}
            for i in range(5)
        ]
        count = bulk_create_predictions(db_session, items)
        assert count == 5

    def test_empty_list_returns_zero(self, db_session) -> None:
        assert bulk_create_predictions(db_session, []) == 0


class TestCountPredictionsByService:
    def test_counts_specific_service(self, db_session) -> None:
        for _ in range(2):
            create_prediction(db_session, {
                "service_name": "target", "cpu_util": 0.5, "mem_util": 0.4,
                "latency_ms": 10.0, "error_rate": 0.0, "throughput": 100.0,
                "predicted_label": "normal", "confidence": 0.9,
            })
        create_prediction(db_session, {
            "service_name": "other", "cpu_util": 0.5, "mem_util": 0.4,
            "latency_ms": 10.0, "error_rate": 0.0, "throughput": 100.0,
            "predicted_label": "normal", "confidence": 0.9,
        })
        assert count_predictions_by_service(db_session, "target") == 2

    def test_zero_for_unknown_service(self, db_session) -> None:
        assert count_predictions_by_service(db_session, "nonexistent") == 0
